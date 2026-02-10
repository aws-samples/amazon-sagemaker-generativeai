"""
DJL Inference Handler for Fine-tuned Chronos-2 Model
Supports S3 paths for context and future dataframes
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import pandas as pd
import torch
from djl_python import Input, Output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chronos2Handler:
    """Handler for Chronos-2 inference with S3 data loading support."""
    
    def __init__(self):
        self.pipeline = None
        self.device = None
        self.s3_client = None
        self.initialized = False
    
    def initialize(self, properties: Dict[str, Any]):
        """Initialize the Chronos-2 pipeline from model artifacts."""
        from chronos import BaseChronosPipeline, Chronos2Pipeline
        
        # Get model path from properties
        model_id = properties.get("model_id", "/opt/ml/model")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load the fine-tuned model
        logger.info(f"Loading Chronos-2 model from: {model_id}")
        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=self.device
        )
        
        # Initialize S3 client
        self.s3_client = boto3.client("s3")
        
        self.initialized = True
        logger.info("Chronos-2 model initialized successfully")
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse S3 URI into bucket and key."""
        s3_uri = s3_uri.replace("s3://", "")
        parts = s3_uri.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        return bucket, key
    
    def _load_df_from_s3(self, s3_uri: str) -> pd.DataFrame:
        """Load DataFrame from S3 (supports parquet and csv)."""
        bucket, key = self._parse_s3_uri(s3_uri)
        
        # Download to temp file
        local_path = f"/tmp/{os.path.basename(key)}"
        self.s3_client.download_file(bucket, key, local_path)
        
        # Load based on extension
        if key.endswith(".parquet"):
            df = pd.read_parquet(local_path)
        elif key.endswith(".csv"):
            df = pd.read_csv(local_path)
        else:
            raise ValueError(f"Unsupported file format: {key}")
        
        # Cleanup
        os.remove(local_path)
        return df

    def _load_df_from_input(self, data: Any) -> pd.DataFrame:
        """Load DataFrame from input (S3 URI, dict, or list)."""
        if isinstance(data, str) and data.startswith("s3://"):
            return self._load_df_from_s3(data)
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def inference(self, inputs: Input) -> Output:
        """
        Run inference on input data.
        
        Expected input JSON format:
        {
            "context_data": "s3://bucket/path/to/context.parquet" or {...} or [...],
            "future_data": "s3://bucket/path/to/future.parquet" or {...} or [...],  # optional
            "parameters": {
                "prediction_length": 24,
                "quantile_levels": [0.1, 0.5, 0.9],
                "id_column": "id",
                "timestamp_column": "timestamp",
                "target": "target_column_name"
            }
        }
        """
        output = Output()
        
        try:
            # Parse input
            input_data = inputs.get_as_json()
            logger.info(f"Received input keys: {input_data.keys()}")
            
            # Extract parameters
            params = input_data.get("parameters", {})
            prediction_length = params.get("prediction_length", 24)
            quantile_levels = params.get("quantile_levels", [0.1, 0.5, 0.9])
            id_column = params.get("id_column", "id")
            timestamp_column = params.get("timestamp_column", "timestamp")
            target = params.get("target", "target")
            
            # Load context data
            context_data = input_data.get("context_data")
            if context_data is None:
                raise ValueError("context_data is required")
            
            context_df = self._load_df_from_input(context_data)
            logger.info(f"Loaded context_df: {context_df.shape}")
            
            # Ensure timestamp is datetime
            if timestamp_column in context_df.columns:
                context_df[timestamp_column] = pd.to_datetime(context_df[timestamp_column])
            
            # Load future data (optional)
            future_data = input_data.get("future_data")
            future_df = None
            if future_data is not None:
                future_df = self._load_df_from_input(future_data)
                if timestamp_column in future_df.columns:
                    future_df[timestamp_column] = pd.to_datetime(future_df[timestamp_column])
                logger.info(f"Loaded future_df: {future_df.shape}")
            
            # Run inference
            logger.info(f"Running inference with prediction_length={prediction_length}")
            pred_df = self.pipeline.predict_df(
                context_df,
                future_df=future_df,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                id_column=id_column,
                timestamp_column=timestamp_column,
                target=target,
            )
            
            # Convert predictions to JSON-serializable format
            pred_df[timestamp_column] = pred_df[timestamp_column].astype(str)
            predictions = pred_df.to_dict(orient="records")
            
            response = {
                "predictions": predictions,
                "prediction_length": prediction_length,
                "num_series": pred_df[id_column].nunique() if id_column in pred_df.columns else 1,
                "status": "success"
            }
            
            output.add_as_json(response)
            logger.info(f"Inference completed: {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}", exc_info=True)
            output.add_as_json({
                "status": "error",
                "error": str(e)
            })
            output.set_code(500)
        
        return output


# Global handler instance
_handler = Chronos2Handler()


def handle(inputs: Input) -> Optional[Output]:
    """DJL entry point for inference."""
    if not _handler.initialized:
        properties = inputs.get_properties()
        _handler.initialize(properties)
    
    if inputs.is_empty():
        return None
    
    return _handler.inference(inputs)
