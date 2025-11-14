from __future__ import annotations

import yaml
from pydantic import BaseModel, model_validator


class Config(BaseModel, extra="forbid", validate_assignment=True):
    test_dataset: str | None = None
    source_model_path: str | None = None
    source_model_predictions_path: str | None = None
    target_model_path: str | None = None
    target_model_predictions_path: str | None = None
    reference_model: str | None = None
    reference_model_predictions_path: str | None = None
    mlflow_tracking_server_arn: str
    mlflow_experiment_name: str
    mlflow_run_name: str
    metrics: list | None = None
    model_judge: str | None = None
    model_judge_parameters: dict | None = None
    custom_metrics: list | None = None

    @model_validator(mode="after")
    def check_model_paths(self) -> Config:
        if not (self.source_model_path or self.source_model_predictions_path):
            raise ValueError("You must provide either 'source_model_path' or 'source_model_predictions_path'")

        if not (self.target_model_path or self.target_model_predictions_path):
            raise ValueError("You must provide either 'target_model_path' or 'target_model_predictions_path'")

        # If predictions are NOT provided for any model,
        # test_dataset is required
        predictions_missing = not all([
            self.source_model_predictions_path,
            self.target_model_predictions_path,
        ])
        if predictions_missing and not self.test_dataset:
            raise ValueError("If predictions are not available, 'test_dataset' is required.")

        return self

    @classmethod
    def load(cls, config_path: str) -> Config:
        raw_config = cls.load_yaml(config_path)
        return cls(
            test_dataset=raw_config.get("test_dataset"),
            source_model_path=raw_config.get("source_model_path"),
            source_model_predictions_path=raw_config.get("source_model_predictions_path"),
            target_model_path=raw_config.get("target_model_path"),
            target_model_predictions_path=raw_config.get("target_model_predictions_path"),
            reference_model=raw_config.get("reference_model"),
            reference_model_predictions_path=raw_config.get("reference_model_predictions_path"),
            mlflow_tracking_server_arn=raw_config["mlflow_tracking_server_arn"],
            mlflow_experiment_name=raw_config["mlflow_experiment_name"],
            mlflow_run_name=raw_config["mlflow_run_name"],
            metrics=raw_config["metrics"],
            model_judge=raw_config.get("model_judge"),
            model_judge_parameters=raw_config.get("model_judge_parameters"),
            custom_metrics=raw_config.get("custom_metrics")
    )

    @staticmethod
    def load_yaml(path: str) -> dict:
        with open(path) as stream:
            return yaml.safe_load(stream)
