"""Small CPU neural networks plus classical and experimental regression models."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class Autoencoder(BaseEstimator):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None):
        normal = X if y is None else X[np.asarray(y) == 0]
        self.network_ = MLPRegressor(
            hidden_layer_sizes=(32, 8, 32),
            activation="relu",
            max_iter=80,
            early_stopping=True,
            random_state=self.random_state,
            batch_size=256,
        )
        self.network_.fit(normal, normal)
        return self

    def decision_function(self, X):
        return np.mean((X - self.network_.predict(X)) ** 2, axis=1)


def preprocessor(x):
    num = x.select_dtypes(include="number").columns.tolist()
    cat = x.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                ),
                num,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
                cat,
            ),
        ]
    )


def candidates():
    return {
        "Dummy baseline": ("classifier", DummyClassifier(strategy="prior")),
        "Logistic Regression": (
            "classifier",
            LogisticRegression(class_weight="balanced", max_iter=700, random_state=42),
        ),
        "KNN": (
            "classifier",
            KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=2),
        ),
        "Decision Tree": (
            "classifier",
            DecisionTreeClassifier(
                max_depth=7,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Random Forest": (
            "classifier",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=2,
                random_state=42,
            ),
        ),
        "Gradient Boosting": (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=15,
                l2_regularization=2,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Deep Neural Network": (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=12,
                batch_size=256,
                random_state=42,
            ),
        ),
        "Linear Regression": ("regression", LinearRegression()),
        "Polynomial Regression": (
            "regression",
            make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=10)
            ),
        ),
        "SVR": ("regression", SVR(C=1, epsilon=0.05, cache_size=400)),
        "Isolation Forest": (
            "anomaly",
            IsolationForest(n_estimators=100, random_state=42, n_jobs=2),
        ),
        "One-Class SVM": (
            "anomaly",
            OneClassSVM(nu=0.02, gamma="scale", cache_size=400),
        ),
        "Autoencoder": ("autoencoder", Autoencoder()),
    }


def score(model, x, kind):
    if kind == "classifier":
        return model.predict_proba(x)[:, 1]
    if kind == "regression":
        return model.predict(x)  # Unclipped: these are NOT probabilities.
    if kind == "anomaly":
        return -model.decision_function(x)
    return model.decision_function(x)
