"""Simple loop-based hyperparameter search for the thread transformer.

The grid is hardcoded — chosen to vary the few knobs that actually move the
needle on this task (frozen encoder, ~7k threads):

    lr        : 1e-4 | 3e-4 | 1e-3        (optimisation step size)
    dropout   : 0.1  | 0.3                (regularisation, helps small data)
    n_layers  : 2    | 4                  (depth vs overfit)

d_model, batch_size, max_comments, etc. are kept at the train-mode defaults
because they do not move the val metric meaningfully on this size of data.

Best model (by val AUC) is saved to the same checkpoint path as `--mode train`.
A sweep summary JSON is written next to it.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import get_linear_schedule_with_warmup

from .model import TransitionAwareThreadTransformer
from .trainer import ThreadTrainer


GRID: Dict[str, List] = {
    "lr": [1e-4, 3e-4, 1e-3],
    "dropout": [0.1, 0.3],
    "n_layers": [2, 4],
}


def _iter_grid():
    keys = list(GRID.keys())
    for values in itertools.product(*[GRID[k] for k in keys]):
        yield dict(zip(keys, values))


def run_sweep(
    variant: str,
    args,
    encoder,
    device: torch.device,
    splits: Dict[str, list],
    make_loader,
    ckpt_path: Path,
) -> Dict:
    train_loader = make_loader(splits["train"], args.batch_size, shuffle=True)
    val_loader = make_loader(splits["val"], args.batch_size, shuffle=False)
    test_loader = make_loader(splits["test"], args.batch_size, shuffle=False)

    encoder_dim = encoder.OUTPUT_DIM if variant == "standalone" else encoder.output_dim

    history: List[Dict] = []
    best_val_auc = -float("inf")
    best_trial: Dict = {}
    best_state = None

    trials = list(_iter_grid())
    print(f"[sweep] {len(trials)} trials over {list(GRID.keys())}")

    for trial_idx, hp in enumerate(trials, start=1):
        print(f"\n[sweep] === trial {trial_idx}/{len(trials)} : {hp} ===")

        model = TransitionAwareThreadTransformer(
            comment_dim=encoder_dim,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=hp["n_layers"],
            ffn_dim=args.ffn_dim,
            dropout=hp["dropout"],
            max_comments=args.max_comments,
            use_discourse_acts=(variant == "with_discourse"),
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=hp["lr"], weight_decay=args.weight_decay
        )
        total_steps = max(len(train_loader) * args.epochs, 1)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_frac),
            num_training_steps=total_steps,
        )
        trainer = ThreadTrainer(
            encoder=encoder, model=model, device=device,
            optimizer=optimizer, scheduler=scheduler,
        )
        trainer.fit(
            train_loader, val_loader,
            epochs=args.epochs,
            early_stop_metric="auc",
            early_stop_patience=args.early_stop_patience,
        )
        # `fit` restores best-by-val weights into `model` before returning.
        val_metrics = trainer.evaluate(val_loader)
        test_metrics = trainer.evaluate(test_loader)

        record = {
            "trial": trial_idx,
            "hp": hp,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(record)
        print(f"[sweep] trial {trial_idx}: val_auc={val_metrics['auc']:.4f}  "
              f"test_auc={test_metrics['auc']:.4f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_trial = record
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[sweep] new best: val_auc={best_val_auc:.4f}")

        del model, optimizer, scheduler, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[sweep] === finished ===")
    print(f"[sweep] best trial: {best_trial['trial']}  hp={best_trial['hp']}")
    print(f"[sweep] best val: {json.dumps(best_trial['val'], indent=2)}")
    print(f"[sweep] best test: {json.dumps(best_trial['test'], indent=2)}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    saved_args = copy.deepcopy(vars(args))
    saved_args.update({
        "n_layers": best_trial["hp"]["n_layers"],
        "dropout": best_trial["hp"]["dropout"],
        "lr": best_trial["hp"]["lr"],
    })
    torch.save(
        {
            "variant": variant,
            "model_state_dict": best_state,
            "args": saved_args,
            "best_hp": best_trial["hp"],
            "val_metrics": best_trial["val"],
            "test_metrics": best_trial["test"],
        },
        ckpt_path,
    )
    print(f"[sweep] saved best checkpoint to {ckpt_path}")

    summary_path = ckpt_path.with_suffix(".sweep.json")
    summary = {
        "variant": variant,
        "grid": GRID,
        "best_trial": best_trial,
        "history": history,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sweep] saved sweep summary to {summary_path}")

    return summary
