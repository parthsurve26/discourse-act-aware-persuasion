from __future__ import annotations

from typing import Iterable, List

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer


def text_baseline_pipeline(max_features: int = 50000) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=max_features,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )


def text_plus_numeric_pipeline(
    text_col: str = "input_text",
    numeric_cols: List[str] | None = None,
    max_features: int = 50000,
) -> Pipeline:
    numeric_cols = numeric_cols or ["num_chars", "num_words"]

    from sklearn.pipeline import FeatureUnion

    text_branch = Pipeline(
        steps=[
            ("select", FunctionTransformer(lambda df: df[text_col], validate=False)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=2)),
        ]
    )
    numeric_branch = Pipeline(
        steps=[
            ("select", FunctionTransformer(lambda df: df[numeric_cols], validate=False)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=2), text_col),
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler(with_mean=False)),
                                ]
                            ),
                            numeric_cols,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

