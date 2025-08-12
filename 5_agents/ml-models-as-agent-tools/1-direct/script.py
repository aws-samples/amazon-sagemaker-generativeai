#!/usr/bin/env python
# coding: utf-8

"""
XGBoost training and inference script for SageMaker

This script is used by SageMaker to train an XGBoost model for demand forecasting
and to serve predictions from a deployed endpoint.
"""

import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging
import io

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
    """Parse SageMaker training job arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--objective', type=str, default='reg:squarederror')
    parser.add_argument('--num_round', type=int, default=100)

    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # Model directory: this is where the model will be saved
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_args()


def load_data(data_dir):
    """Load training data from CSV file."""
    logger.info(f"Loading data from {data_dir}")
    
    # List files in the directory
    files = os.listdir(data_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Load the first CSV file
    data_path = os.path.join(data_dir, csv_files[0])
    df = pd.read_csv(data_path)
    
    # Separate features and target
    if 'demand' in df.columns:
        y = df['demand']
        X = df.drop(['demand'], axis=1)
    else:
        # If 'demand' column is not present, assume the last column is the target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    
    return X, y


def train(args):
    """Train XGBoost model with the given arguments."""
    logger.info("Loading training data")
    X_train, y_train = load_data(args.train)
    
    logger.info("Loading validation data")
    X_val, y_val = load_data(args.validation)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set XGBoost parameters
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective
    }
    
    # Train model
    logger.info("Training XGBoost model")
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=10
    )
    
    # Save the model
    logger.info(f"Saving model to {args.model_dir}")
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_path)
    
    # Save feature names for inference
    feature_names = X_train.columns.tolist()
    with open(os.path.join(args.model_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    return model


def model_fn(model_dir):
    """Load the XGBoost model for inference."""
    # Load the XGBoost model
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    # Return both model and feature names
    return {'model': model, 'feature_names': feature_names}


def input_fn(request_body, request_content_type):
    """Parse input data for prediction."""
    if request_content_type == 'text/csv':
        # Parse CSV input
        data = io.StringIO(request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body)
        df = pd.read_csv(data, header=None)
        return df
    elif request_content_type == 'application/json':
        # Parse JSON input
        json_data = json.loads(request_body.decode('utf-8') if isinstance(request_body, bytes) else request_body)
        # Handle both list of lists and dict with features
        if isinstance(json_data, list):
            df = pd.DataFrame(json_data)
        else:
            df = pd.DataFrame([json_data])
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Use 'text/csv' or 'application/json'.")


def predict_fn(input_data, model_dict):
    """Make predictions using the loaded model."""
    # Extract model and feature names
    model = model_dict['model']
    feature_names = model_dict['feature_names']
    
    # Ensure input data has the correct columns/order
    if len(input_data.columns) != len(feature_names):
        raise ValueError(f"Input data has {len(input_data.columns)} features, but model expects {len(feature_names)}")
    
    # Convert to DMatrix for prediction
    dmatrix = xgb.DMatrix(input_data.values)
    
    # Make prediction
    predictions = model.predict(dmatrix)
    
    return predictions


def output_fn(predictions, content_type):
    """Format predictions for response."""
    if content_type == 'application/json':
        # Convert predictions to a list and return as JSON
        predictions_list = predictions.tolist()
        return json.dumps(predictions_list)
    else:
        raise ValueError(f"Unsupported accept type: {content_type}. Use 'application/json'.")


if __name__ == '__main__':
    args = parse_args()
    model = train(args)
    logger.info("Training completed successfully")