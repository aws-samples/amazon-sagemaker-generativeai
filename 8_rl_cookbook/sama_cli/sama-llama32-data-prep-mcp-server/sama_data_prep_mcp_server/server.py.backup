#!/usr/bin/env python3
"""
SAMA Data Preparation MCP Server - Q CLI Compatible
Provides tools for dataset preparation and formatting for SAMA agents
Supports instruction-following, summarization, and QA dataset preparation
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from datasets import load_dataset
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from sagemaker.s3 import S3Uploader
import sagemaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastMCP server for SAMA
mcp = FastMCP('sama-data-prep-mcp-server')


def get_sagemaker_session():
    """Get SageMaker session with error handling."""
    try:
        return sagemaker.Session()
    except NoCredentialsError:
        raise Exception("AWS credentials not configured")
    except Exception as e:
        raise Exception(f"Failed to create SageMaker session: {str(e)}")


@mcp.tool(description="Prepare a generic instruction-tuning dataset (e.g., Dolly) with optional category filtering")
async def prepare_instruction_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name (e.g., databricks/databricks-dolly-15k)"),
    category: Optional[str] = Field(default=None, description="Filter by category (e.g., 'summarization', 'closed_qa')"),
    category_field: str = Field(default="category", description="Field name for category filtering"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data")
) -> Dict[str, Any]:
    """
    Load an instruction-following dataset and prepare it for fine-tuning.
    If `category` is provided, filters the dataset by that category.
    Splits into train/test and saves the train split as JSONL.
    """
    try:
        logger.info(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")
        
        # Filter by category if specified
        if category is not None:
            logger.info(f"Filtering dataset for category = {category}")
            ds = ds.filter(lambda example: example.get(category_field) == category)
            # Remove the category column if present
            if category_field in ds.column_names:
                ds = ds.remove_columns(category_field)
        
        # Split into train and test
        if test_size and test_size > 0:
            logger.info(f"Splitting dataset into train/test with test_size = {test_size}")
            data_splits = ds.train_test_split(test_size=test_size)
            train_ds = data_splits["train"]
            test_ds = data_splits["test"]
        else:
            train_ds = ds
            test_ds = None

        # Ensure output directory exists
        output_path = Path(local_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_file = output_path / "train.jsonl"
        
        logger.info(f"Saving training data to {train_file}")
        train_ds.to_json(str(train_file))

        # Example first training sample (for verification)
        sample_data = train_ds[0] if len(train_ds) > 0 else None

        return {
            "status": "success",
            "message": f"Dataset prepared (instruction format){' for category ' + category if category else ''}",
            "train_file": str(train_file),
            "train_samples": len(train_ds),
            "test_samples": len(test_ds) if test_ds is not None else 0,
            "sample_data": sample_data,
            "dataset_columns": train_ds.column_names if train_ds else []
        }
    except Exception as e:
        logger.error(f"Error preparing instruction dataset: {e}")
        return {"status": "error", "message": f"Failed to prepare dataset: {str(e)}"}


@mcp.tool(description="Prepare a summarization dataset by formatting it into instruction+context->response format")
async def prepare_summarization_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name for summarization"),
    text_field: Optional[str] = Field(default=None, description="Field containing text to summarize"),
    summary_field: Optional[str] = Field(default=None, description="Field containing summary"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data")
) -> Dict[str, Any]:
    """
    Prepare a summarization dataset for fine-tuning. 
    The dataset should contain an article/text and its summary. 
    Converts each example into an instruction format.
    """
    try:
        logger.info(f"Loading summarization dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")
        
        # Try to infer text and summary field names if not provided
        if text_field is None or summary_field is None:
            cols = ds.column_names
            # Common field names for summarization datasets
            text_candidates = ["article", "document", "dialogue", "text"]
            summary_candidates = ["summary", "highlights"]
            
            # Guess text_field
            for col in text_candidates:
                if col in cols:
                    text_field = text_field or col
                    break
            
            # Guess summary_field
            for col in summary_candidates:
                if col in cols:
                    summary_field = summary_field or col
                    break
                    
            if text_field is None or summary_field is None:
                raise ValueError("Could not infer text/summary fields. Please specify them explicitly.")
        
        logger.info(f"Using text field = '{text_field}', summary field = '{summary_field}'")

        # Map dataset to instruction format
        def map_to_instruction(example):
            instruction = "Summarize the following text."
            context = example[text_field]
            response = example[summary_field]
            return {"instruction": instruction, "context": context, "response": response}
        
        ds = ds.map(map_to_instruction, remove_columns=[text_field, summary_field])
        
        # Split into train and test sets if requested
        if test_size and test_size > 0:
            logger.info(f"Splitting dataset into train/test with test_size = {test_size}")
            data_splits = ds.train_test_split(test_size=test_size)
            train_ds = data_splits["train"]
            test_ds = data_splits["test"]
        else:
            train_ds = ds
            test_ds = None

        # Save train split to JSONL
        output_path = Path(local_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_file = output_path / "train.jsonl"
        
        logger.info(f"Saving formatted training data to {train_file}")
        train_ds.to_json(str(train_file))

        sample_data = train_ds[0] if len(train_ds) > 0 else None
        
        return {
            "status": "success",
            "message": "Summarization dataset formatted and saved",
            "train_file": str(train_file),
            "train_samples": len(train_ds),
            "test_samples": len(test_ds) if test_ds is not None else 0,
            "sample_data": sample_data,
            "dataset_columns": train_ds.column_names if train_ds else []
        }
    except Exception as e:
        logger.error(f"Error preparing summarization dataset: {e}")
        return {"status": "error", "message": f"Failed to prepare summarization data: {str(e)}"}


@mcp.tool(description="Prepare a QA dataset (with question, answer, and optional context) in instruction format")
async def prepare_qa_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name for QA"),
    context_field: Optional[str] = Field(default=None, description="Field containing context/passage"),
    question_field: str = Field(default="question", description="Field containing questions"),
    answer_field: str = Field(default="answers", description="Field containing answers"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data")
) -> Dict[str, Any]:
    """
    Prepare a Question-Answering dataset for fine-tuning.
    Expects a question and an answer for each example, and optionally a context/passage.
    Formats each example as an instruction (the question) with context and the answer as response.
    """
    try:
        logger.info(f"Loading QA dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")
        
        # If context_field not specified, try to infer
        if context_field is None:
            if "context" in ds.column_names:
                context_field = "context"
            elif "paragraph" in ds.column_names:
                context_field = "paragraph"
            else:
                context_field = None  # no context in dataset (open QA scenario)
        
        logger.info(f"Using context field = '{context_field}', question field = '{question_field}', answer field = '{answer_field}'")

        # Function to map each QA example to instruction format
        def map_to_instruction(example):
            instruction = example[question_field]  # use the question as the instruction prompt
            context = example[context_field] if context_field and context_field in example else ""
            
            # Extract answer text
            answer = example[answer_field]
            answer_text = ""
            
            if answer is None:
                answer_text = ""
            elif isinstance(answer, str):
                answer_text = answer
            elif isinstance(answer, dict) and "text" in answer:
                # Many QA datasets (e.g. SQuAD) have answers as a dict with a 'text' list
                texts = answer.get("text")
                if texts:
                    answer_text = texts[0] if isinstance(texts, list) else texts
            elif isinstance(answer, list):
                # If it's a list of answers (strings), take the first element
                answer_text = answer[0] if len(answer) > 0 else ""
            else:
                # Unrecognized format, just cast to string
                answer_text = str(answer)
            
            response = answer_text
            return {"instruction": instruction, "context": context, "response": response}
        
        # Map and format the dataset to new columns
        remove_cols = [question_field]
        if context_field and context_field in ds.column_names:
            remove_cols.append(context_field)
        if answer_field in ds.column_names:
            remove_cols.append(answer_field)
        
        ds = ds.map(map_to_instruction, remove_columns=remove_cols)

        # Split into train/test if needed
        if test_size and test_size > 0:
            logger.info(f"Splitting dataset into train/test with test_size = {test_size}")
            data_splits = ds.train_test_split(test_size=test_size)
            train_ds = data_splits["train"]
            test_ds = data_splits["test"]
        else:
            train_ds = ds
            test_ds = None

        # Save training data to JSONL
        output_path = Path(local_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_file = output_path / "train.jsonl"
        
        logger.info(f"Saving QA training data to {train_file}")
        train_ds.to_json(str(train_file))

        sample_data = train_ds[0] if len(train_ds) > 0 else None
        
        return {
            "status": "success",
            "message": "QA dataset formatted and saved",
            "train_file": str(train_file),
            "train_samples": len(train_ds),
            "test_samples": len(test_ds) if test_ds is not None else 0,
            "sample_data": sample_data,
            "dataset_columns": train_ds.column_names if train_ds else []
        }
    except Exception as e:
        logger.error(f"Error preparing QA dataset: {e}")
        return {"status": "error", "message": f"Failed to prepare QA data: {str(e)}"}


@mcp.tool(description="Create an instruction prompt template JSON file for fine-tuning")
async def create_instruction_template(
    instruction_field: str = Field(default="instruction", description="Field name for instructions"),
    context_field: str = Field(default="context", description="Field name for context/input"),
    response_field: str = Field(default="response", description="Field name for responses"),
    output_path: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep/template.json", description="Path to save template file")
) -> Dict[str, Any]:
    """
    Create a template JSON file that defines how to format the prompt and completion for training.
    The template uses placeholders for the instruction, context, and response.
    """
    try:
        logger.info("Creating instruction template")
        
        # Template as used in SageMaker JumpStart fine-tuning examples
        template = {
            "prompt": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{{{instruction_field}}}\n\n### Input:\n{{{context_field}}}\n\n"
            ),
            "completion": f" {{{response_field}}}"
        }
        
        # Save template to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        return {
            "status": "success",
            "message": "Instruction template created",
            "template_path": str(output_file),
            "template": template
        }
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        return {"status": "error", "message": f"Failed to create template: {str(e)}"}


@mcp.tool(description="Upload local training data (and template) to S3 for SageMaker training")
async def upload_training_data_to_s3(
    local_data_file: str = Field(description="Path to local training data file (e.g., train.jsonl)"),
    template_file: Optional[str] = Field(default=None, description="Path to template file (optional)"),
    s3_prefix: str = Field(default="finetune_dataset", description="S3 prefix for uploaded files"),
    bucket_name: Optional[str] = Field(default=None, description="S3 bucket name (uses default SageMaker bucket if not specified)")
) -> Dict[str, Any]:
    """
    Upload the prepared training data (and optional template file) to an S3 bucket/prefix.
    If bucket_name is not provided, uses the default SageMaker session bucket.
    """
    try:
        logger.info("Uploading training data to S3")
        sagemaker_session = get_sagemaker_session()
        
        # Determine target bucket
        if bucket_name is None:
            bucket_name = sagemaker_session.default_bucket()
        
        # Determine default prefix (if any configured in SageMaker session)
        default_prefix = sagemaker_session.default_bucket_prefix or ""
        
        # Construct S3 URI
        base_path = f"s3://{bucket_name}"
        if default_prefix:
            base_path = f"{base_path}/{default_prefix.rstrip('/')}"
        target_path = f"{base_path}/{s3_prefix}"
        
        # Upload files
        uploaded_files = []
        
        # Upload training data
        if not os.path.exists(local_data_file):
            raise FileNotFoundError(f"Training data file not found: {local_data_file}")
        
        result = S3Uploader.upload(local_data_file, target_path)
        uploaded_files.append(local_data_file)
        logger.info(f"Uploaded {local_data_file} to {target_path}")
        
        # Upload template if provided
        if template_file and os.path.exists(template_file):
            S3Uploader.upload(template_file, target_path)
            uploaded_files.append(template_file)
            logger.info(f"Uploaded {template_file} to {target_path}")
        
        return {
            "status": "success",
            "message": "Files uploaded to S3",
            "s3_path": target_path,
            "bucket": bucket_name,
            "uploaded_files": uploaded_files
        }
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return {"status": "error", "message": f"Failed to upload to S3: {str(e)}"}


@mcp.tool(description="Get dataset information and preview samples")
async def inspect_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name to inspect"),
    split: str = Field(default="train", description="Dataset split to inspect"),
    num_samples: int = Field(default=3, description="Number of sample records to show")
) -> Dict[str, Any]:
    """
    Inspect a HuggingFace dataset to understand its structure and content.
    Useful for understanding field names and data format before preparation.
    """
    try:
        logger.info(f"Inspecting dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split=split)
        
        # Get basic info
        info = {
            "dataset_name": dataset_name,
            "split": split,
            "num_rows": len(ds),
            "column_names": ds.column_names,
            "features": {name: str(feature) for name, feature in ds.features.items()},
            "samples": []
        }
        
        # Get sample records
        for i in range(min(num_samples, len(ds))):
            info["samples"].append(ds[i])
        
        return {
            "status": "success",
            "message": f"Dataset inspection complete",
            "dataset_info": info
        }
    except Exception as e:
        logger.error(f"Error inspecting dataset: {e}")
        return {"status": "error", "message": f"Failed to inspect dataset: {str(e)}"}


@mcp.tool(description="Perform full data preparation workflow for a specific dataset")
async def full_data_prep_workflow(
    dataset_name: str = Field(description="HuggingFace dataset name (e.g., databricks/databricks-dolly-15k)"),
    dataset_type: str = Field(default="instruction", description="Type of dataset: 'instruction', 'summarization', or 'qa'"),
    category: Optional[str] = Field(default=None, description="Category filter for instruction datasets"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    upload_to_s3: bool = Field(default=True, description="Whether to upload prepared data to S3"),
    s3_prefix: str = Field(default="finetune_dataset", description="S3 prefix for uploaded files"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data")
) -> Dict[str, Any]:
    """
    Perform a complete data preparation workflow:
    1. Inspect the dataset
    2. Prepare the data based on type
    3. Create instruction template
    4. Optionally upload to S3
    """
    try:
        logger.info(f"Starting full data prep workflow for {dataset_name}")
        results = {"steps": []}
        
        # Step 1: Inspect dataset
        logger.info("Step 1: Inspecting dataset")
        inspect_result = await inspect_dataset(dataset_name, num_samples=2)
        results["steps"].append({"step": "inspect", "result": inspect_result})
        
        if inspect_result["status"] != "success":
            return {"status": "error", "message": "Failed at dataset inspection", "results": results}
        
        # Step 2: Prepare data based on type
        logger.info(f"Step 2: Preparing {dataset_type} dataset")
        if dataset_type == "instruction":
            prep_result = await prepare_instruction_dataset(
                dataset_name=dataset_name,
                category=category,
                test_size=test_size,
                local_output_dir=local_output_dir
            )
        elif dataset_type == "summarization":
            prep_result = await prepare_summarization_dataset(
                dataset_name=dataset_name,
                test_size=test_size,
                local_output_dir=local_output_dir
            )
        elif dataset_type == "qa":
            prep_result = await prepare_qa_dataset(
                dataset_name=dataset_name,
                test_size=test_size,
                local_output_dir=local_output_dir
            )
        else:
            return {"status": "error", "message": f"Unsupported dataset type: {dataset_type}"}
        
        results["steps"].append({"step": "prepare", "result": prep_result})
        
        if prep_result["status"] != "success":
            return {"status": "error", "message": "Failed at data preparation", "results": results}
        
        # Step 3: Create template
        logger.info("Step 3: Creating instruction template")
        template_path = os.path.join(local_output_dir, "template.json")
        template_result = await create_instruction_template(output_path=template_path)
        results["steps"].append({"step": "template", "result": template_result})
        
        # Step 4: Upload to S3 if requested
        if upload_to_s3:
            logger.info("Step 4: Uploading to S3")
            train_file = prep_result.get("train_file")
            if train_file:
                upload_result = await upload_training_data_to_s3(
                    local_data_file=train_file,
                    template_file=template_path if template_result["status"] == "success" else None,
                    s3_prefix=s3_prefix
                )
                results["steps"].append({"step": "upload", "result": upload_result})
        
        return {
            "status": "success",
            "message": "Full data preparation workflow completed",
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in full data prep workflow: {e}")
        return {"status": "error", "message": f"Workflow failed: {str(e)}", "results": results}


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="SAMA Data Preparation MCP Server")
    parser.add_argument("--port", type=int, default=3001, help="Port to run the server on")
    parser.add_argument("--host", default="localhost", help="Host to run the server on")
    
    args = parser.parse_args()
    
    logger.info("Starting SAMA Data Preparation MCP Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
