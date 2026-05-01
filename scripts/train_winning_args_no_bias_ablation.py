"""No-bias ablation: with_discourse variant trained with speaker + distance bias disabled.

Isolates the contribution of the act_pair_bias by removing the other two
transition signals. Saves to a separate checkpoint so it does not clobber
the main with_discourse run.
"""
import sys

from _bootstrap import *  # noqa: F401,F403
from discourse_act_aware_persuasion.winning_arguments.cli import run


if __name__ == "__main__":
    # Best hyperparams from with_discourse sweep, held fixed for fair ablation comparison.
    extra = [
        "--no-speaker-bias", "--no-distance-bias",
        "--lr", "0.0001",
        "--dropout", "0.1",
        "--n-layers", "4",
    ]
    if "--ckpt-path" not in sys.argv:
        extra += ["--ckpt-path", "data/models/winning_args_with_discourse_no_bias.pt"]
    run(variant="with_discourse", argv=sys.argv[1:] + extra)
