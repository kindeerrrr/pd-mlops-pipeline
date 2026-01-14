import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay
)


DATA_PATH = "data/processed/clean_data.csv"
TARGET = "target"

def load_data():
    return pd.read_csv(DATA_PATH)

def train_model():
    mlflow.set_experiment("pd_default_prediction")

    with mlflow.start_run():
        df = load_data()

        X = df.drop(columns=[TARGET])
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000))
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("scaler", "StandardScaler")

        mlflow.sklearn.log_model(pipeline, "model")

        print("Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.savefig("models/roc_curve.png")


if __name__ == "__main__":
    train_model()
