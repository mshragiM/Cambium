"""Model training and evaluation."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multioutput import MultiOutputRegressor

from .config import CambiumConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self, config: CambiumConfig):
        self.config = config
        self.model = None
        self.training_metrics: Dict[str, float] = {}

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> Any:
        try:
            model_config = self.config.config["model"]
            if model_config["type"] != "random_forest":
                raise ValueError(f"Unsupported model type: {model_config['type']}")

            if model_config["task"] == "regression":
                base_model = RandomForestRegressor(
                    n_estimators=model_config["n_estimators"],
                    max_depth=model_config["max_depth"],
                    max_leaf_nodes=model_config.get("max_leaf_nodes"),
                    random_state=model_config["random_state"],
                )
                self.model = MultiOutputRegressor(base_model) if y_train.shape[1] > 1 else base_model
            else:
                self.model = RandomForestClassifier(
                    n_estimators=model_config["n_estimators"],
                    max_depth=model_config["max_depth"],
                    max_leaf_nodes=model_config.get("max_leaf_nodes"),
                    random_state=model_config["random_state"],
                )

            if y_train.ndim == 2 and y_train.shape[1] == 1:
                y_train = y_train.ravel()

            start_time = time.time()
            self.model.fit(x_train, y_train)
            training_time = time.time() - start_time
            self.training_metrics["training_time"] = training_time
            logger.info("Model trained successfully in %.2f seconds", training_time)
            return self.model
        except Exception as exc:
            logger.error("Error training model: %s", exc)
            raise

    def evaluate_model(self, x_test: np.ndarray, y_test: np.ndarray, target_cols: List[str]) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        if len(x_test) == 0 or len(y_test) == 0:
            logger.warning("Empty test data provided for evaluation")
            return {"inference_time": 0.0, "inference_rate": 0.0}

        start_time = time.time()
        y_pred = self.model.predict(x_test)
        inference_time = time.time() - start_time
        inference_rate = len(x_test) / inference_time if inference_time > 0 else 0.0

        metrics: Dict[str, float] = {
            "inference_time": inference_time,
            "inference_rate": inference_rate,
        }

        if self.config.config["model"]["task"] == "regression":
            if len(target_cols) == 1:
                try:
                    metrics["r2_score"] = r2_score(y_test, y_pred)
                except Exception as exc:
                    logger.warning("R2 score calculation failed: %s", exc)
                    metrics["r2_score"] = 0.0
            else:
                for index, target in enumerate(target_cols):
                    try:
                        metrics[f"r2_{target}"] = r2_score(y_test[:, index], y_pred[:, index])
                    except Exception as exc:
                        logger.warning("R2 score calculation for %s failed: %s", target, exc)
                        metrics[f"r2_{target}"] = 0.0
        else:
            try:
                if y_test.ndim == 2 and y_test.shape[1] == 1:
                    y_test = y_test.ravel()
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
            except Exception as exc:
                logger.warning("Accuracy calculation failed: %s", exc)
                metrics["accuracy"] = 0.0

        self.training_metrics.update(metrics)
        logger.info("Model evaluation completed: %s", metrics)
        return metrics
