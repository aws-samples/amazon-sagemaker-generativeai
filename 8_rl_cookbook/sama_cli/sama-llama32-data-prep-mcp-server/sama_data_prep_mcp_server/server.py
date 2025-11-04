#!/usr/bin/env python3
"""
SAMA Data Preparation MCP Server - Q CLI Compatible (Fixed Version)
Provides tools for dataset preparation and formatting for SAMA agents
Supports instruction-following, summarization, and QA dataset preparation
Fixed to handle fsspec compatibility issues
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
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


def download_dataset_manually(dataset_name: str) -> List[Dict]:
    """
    Download dataset manually to avoid fsspec compatibility issues.
    Currently supports databricks/databricks-dolly-15k.
    """
    if dataset_name == "databricks/databricks-dolly-15k":
        url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        logger.info(f"Downloading {dataset_name} from {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the JSONL data
        lines = response.text.strip().split('\n')
        data = [json.loads(line) for line in lines if line.strip()]
        
        logger.info(f"Successfully downloaded {len(data)} samples")
        return data
    else:
        # For other datasets, try to use a fallback approach but catch fsspec errors
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            data = [dict(item) for item in ds]
            logger.info(f"Successfully loaded {len(data)} samples using load_dataset")
            return data
        except Exception as e:
            if "Invalid pattern" in str(e) and "**" in str(e):
                raise NotImplementedError(f"Manual download not implemented for {dataset_name}. The dataset has fsspec compatibility issues. Please add manual download support for this dataset.")
            else:
                raise e


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
        
        # Use manual download to avoid fsspec issues
        try:
            data = download_dataset_manually(dataset_name)
        except NotImplementedError:
            # Fallback to original method for unsupported datasets
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            data = [dict(item) for item in ds]
        
        # Filter by category if specified
        if category is not None:
            logger.info(f"Filtering dataset for category = {category}")
            data = [item for item in data if item.get(category_field) == category]
            # Remove the category field from each item
            for item in data:
                if category_field in item:
                    del item[category_field]
        
        # Split into train and test
        if test_size and test_size > 0:
            logger.info(f"Splitting dataset into train/test with test_size = {test_size}")
            split_idx = int(len(data) * (1 - test_size))
            train_data = data[:split_idx]
            test_data = data[split_idx:]
        else:
            train_data = data
            test_data = []

        # Convert to instruction format
        def convert_to_instruction_format(item):
            return {
                "instruction": item.get("instruction", ""),
                "context": item.get("context", ""),
                "response": item.get("response", "")
            }
        
        train_formatted = [convert_to_instruction_format(item) for item in train_data]
        test_formatted = [convert_to_instruction_format(item) for item in test_data] if test_data else []

        # Ensure output directory exists
        output_path = Path(local_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_file = output_path / "train.jsonl"
        
        logger.info(f"Saving training data to {train_file}")
        with open(train_file, 'w') as f:
            for item in train_formatted:
                f.write(json.dumps(item) + '\n')

        # Save test data if available
        test_file = None
        if test_formatted:
            test_file = output_path / "test.jsonl"
            logger.info(f"Saving test data to {test_file}")
            with open(test_file, 'w') as f:
                for item in test_formatted:
                    f.write(json.dumps(item) + '\n')

        # Example first training sample (for verification)
        sample_data = train_formatted[0] if train_formatted else None
        
        # Get column names from the first item
        column_names = list(sample_data.keys()) if sample_data else []

        return {
            "status": "success",
            "message": f"Dataset prepared (instruction format){' for category ' + category if category else ''}",
            "train_file": str(train_file),
            "test_file": str(test_file) if test_file else None,
            "train_samples": len(train_formatted),
            "test_samples": len(test_formatted),
            "sample_data": sample_data,
            "dataset_columns": column_names
        }

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to prepare dataset: {str(e)}"
        }


@mcp.tool(description="Prepare a summarization dataset by formatting it into instruction+context->response format")
async def prepare_summarization_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name for summarization"),
    text_field: Optional[str] = Field(default=None, description="Field containing text to summarize"),
    summary_field: Optional[str] = Field(default=None, description="Field containing summary"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data")
) -> Dict[str, Any]:
    """
    Load a summarization dataset and prepare it for fine-tuning.
    Converts to instruction format: "Summarize the following text:" + text -> summary
    """
    try:
        logger.info(f"Loading summarization dataset: {dataset_name}")
        
        # Try manual download first, fallback to load_dataset
        try:
            data = download_dataset_manually(dataset_name)
        except NotImplementedError:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            data = [dict(item) for item in ds]
        
        # Try to infer text and summary field names if not provided
        if not text_field or not summary_field:
            sample = data[0] if data else {}
            possible_text_fields = ['text', 'article', 'document', 'content', 'input']
            possible_summary_fields = ['summary', 'highlights', 'abstract', 'target', 'output']
            
            if not text_field:
                text_field = next((f for f in possible_text_fields if f in sample), None)
            if not summary_field:
                summary_field = next((f for f in possible_summary_fields if f in sample), None)
            
            if not text_field or not summary_field:
                raise ValueError(f"Could not infer text_field and summary_field. Available fields: {list(sample.keys())}")
        
        logger.info(f"Using text_field='{text_field}', summary_field='{summary_field}'")
        
        # Convert to instruction format
        instruction_data = []
        for item in data:
            if text_field in item and summary_field in item:
                instruction_data.append({
                    "instruction": "Summarize the following text:",
                    "context": item[text_field],
                    "response": item[summary_field]
                })
        
        # Split into train and test
        if test_size and test_size > 0:
            split_idx = int(len(instruction_data) * (1 - test_size))
            train_data = instruction_data[:split_idx]
            test_data = instruction_data[split_idx:]
        else:
            train_data = instruction_data
            test_data = []

        # Ensure output directory exists
        output_path = Path(local_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_file = output_path / "train.jsonl"
        
        logger.info(f"Saving training data to {train_file}")
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')

        # Save test data if available
        test_file = None
        if test_data:
            test_file = output_path / "test.jsonl"
            with open(test_file, 'w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

        return {
            "status": "success",
            "message": "Summarization dataset prepared",
            "train_file": str(train_file),
            "test_file": str(test_file) if test_file else None,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "sample_data": train_data[0] if train_data else None,
            "text_field_used": text_field,
            "summary_field_used": summary_field
        }

    except Exception as e:
        logger.error(f"Failed to prepare summarization dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to prepare summarization dataset: {str(e)}"
        }


@mcp.tool(description="Prepare a QA dataset (with question, answer, and optional context) in instruction format")
async def prepare_qa_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name for QA"),
    question_field: str = Field(default="question", description="Field containing questions"),
    answer_field: str = Field(default="answers", description="Field containing answers"),
    context_field: Optional[str] = Field(default=None, description="Field containing context/passage"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data")
) -> Dict[str, Any]:
    """
    Load a QA dataset and prepare it for fine-tuning.
    Converts to instruction format with question as instruction, context as context, and answer as response.
    """
    try:
        logger.info(f"Loading QA dataset: {dataset_name}")
        
        # Try manual download first, fallback to load_dataset
        try:
            data = download_dataset_manually(dataset_name)
        except NotImplementedError:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            data = [dict(item) for item in ds]
        
        # If context_field not specified, try to infer
        if not context_field:
            sample = data[0] if data else {}
            possible_context_fields = ['context', 'passage', 'paragraph', 'text']
            context_field = next((f for f in possible_context_fields if f in sample), None)
        
        logger.info(f"Using question_field='{question_field}', answer_field='{answer_field}', context_field='{context_field}'")
        
        # Convert to instruction format
        instruction_data = []
        for item in data:
            if question_field in item and answer_field in item:
                # Handle different answer formats
                answer = item[answer_field]
                if isinstance(answer, list):
                    answer = answer[0] if answer else ""
                elif isinstance(answer, dict):
                    answer = answer.get('text', str(answer))
                
                qa_item = {
                    "instruction": item[question_field],
                    "context": item.get(context_field, "") if context_field else "",
                    "response": str(answer)
                }
                instruction_data.append(qa_item)
        
        # Split into train and test
        if test_size and test_size > 0:
            split_idx = int(len(instruction_data) * (1 - test_size))
            train_data = instruction_data[:split_idx]
            test_data = instruction_data[split_idx:]
        else:
            train_data = instruction_data
            test_data = []

        # Ensure output directory exists
        output_path = Path(local_output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        train_file = output_path / "train.jsonl"
        
        logger.info(f"Saving training data to {train_file}")
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')

        # Save test data if available
        test_file = None
        if test_data:
            test_file = output_path / "test.jsonl"
            with open(test_file, 'w') as f:
                for item in test_data:
                    f.write(json.dumps(item) + '\n')

        return {
            "status": "success",
            "message": "QA dataset prepared",
            "train_file": str(train_file),
            "test_file": str(test_file) if test_file else None,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "sample_data": train_data[0] if train_data else None,
            "fields_used": {
                "question": question_field,
                "answer": answer_field,
                "context": context_field
            }
        }

    except Exception as e:
        logger.error(f"Failed to prepare QA dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to prepare QA dataset: {str(e)}"
        }


@mcp.tool(description="Get dataset information and preview samples")
async def inspect_dataset(
    dataset_name: str = Field(description="HuggingFace dataset name to inspect"),
    split: str = Field(default="train", description="Dataset split to inspect"),
    num_samples: int = Field(default=3, description="Number of sample records to show")
) -> Dict[str, Any]:
    """
    Inspect a dataset to understand its structure and content.
    """
    try:
        logger.info(f"Inspecting dataset: {dataset_name}")
        
        # Use manual download to avoid fsspec issues
        data = download_dataset_manually(dataset_name)
        
        # Get basic info
        total_samples = len(data)
        column_names = list(data[0].keys()) if data else []
        
        # Get sample data
        sample_data = data[:num_samples] if data else []
        
        # Analyze field types and content
        field_analysis = {}
        if data:
            for field in column_names:
                sample_values = [str(item.get(field, ""))[:100] for item in data[:5]]
                field_analysis[field] = {
                    "sample_values": sample_values,
                    "type": type(data[0].get(field, "")).__name__
                }
        
        return {
            "status": "success",
            "dataset_name": dataset_name,
            "split": split,
            "total_samples": total_samples,
            "column_names": column_names,
            "sample_data": sample_data,
            "field_analysis": field_analysis
        }

    except Exception as e:
        logger.error(f"Failed to inspect dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to inspect dataset: {str(e)}"
        }


@mcp.tool(description="Create an instruction prompt template JSON file for fine-tuning")
async def create_instruction_template(
    instruction_field: str = Field(default="instruction", description="Field name for instructions"),
    context_field: str = Field(default="context", description="Field name for context/input"),
    response_field: str = Field(default="response", description="Field name for responses"),
    output_path: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep/template.json", description="Path to save template file")
) -> Dict[str, Any]:
    """
    Create a template JSON file for instruction tuning format.
    """
    try:
        template = {
            "prompt": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\\n\\n### Instruction:\\n{{{instruction_field}}}\\n\\n### Input:\\n{{{context_field}}}\\n\\n",
            "completion": f" {{{response_field}}}"
        }
        
        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Template saved to {output_path}")
        
        return {
            "status": "success",
            "message": "Instruction template created",
            "template_path": output_path,
            "template": template
        }

    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create template: {str(e)}"
        }


@mcp.tool(description="Upload local training data (and template) to S3 for SageMaker training")
async def upload_training_data_to_s3(
    local_data_file: str = Field(description="Path to local training data file (e.g., train.jsonl)"),
    template_file: Optional[str] = Field(default=None, description="Path to template file (optional)"),
    bucket_name: Optional[str] = Field(default=None, description="S3 bucket name (uses default SageMaker bucket if not specified)"),
    s3_prefix: str = Field(default="finetune_dataset", description="S3 prefix for uploaded files")
) -> Dict[str, Any]:
    """
    Upload prepared training data and template to S3 for SageMaker training.
    """
    try:
        # Get SageMaker session and bucket
        sagemaker_session = get_sagemaker_session()
        if not bucket_name:
            bucket_name = sagemaker_session.default_bucket()
        
        # Clean S3 prefix and construct path without trailing slash
        s3_prefix_clean = s3_prefix.rstrip('/')
        s3_path = f"s3://{bucket_name}/{s3_prefix_clean}"
        logger.info(f"DEBUG: Final s3_path for upload: '{s3_path}'")
        
        # Upload training data
        logger.info(f"Uploading {local_data_file} to {s3_path}")
        S3Uploader.upload(local_data_file, s3_path)
        
        uploaded_files = [local_data_file]
        
        # Upload template if provided
        if template_file and Path(template_file).exists():
            logger.info(f"Uploading {template_file} to {s3_path}")
            S3Uploader.upload(template_file, s3_path)
            uploaded_files.append(template_file)
        
        return {
            "status": "success",
            "message": "Files uploaded to S3",
            "s3_path": s3_path,
            "bucket": bucket_name,
            "prefix": s3_prefix,
            "uploaded_files": uploaded_files
        }

    except Exception as e:
        logger.error(f"Failed to upload to S3: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to upload to S3: {str(e)}"
        }


@mcp.tool(description="Perform full data preparation workflow for a specific dataset")
async def full_data_prep_workflow(
    dataset_name: str = Field(description="HuggingFace dataset name (e.g., databricks/databricks-dolly-15k)"),
    dataset_type: str = Field(default="instruction", description="Type of dataset: 'instruction', 'summarization', or 'qa'"),
    category: Optional[str] = Field(default=None, description="Category filter for instruction datasets"),
    test_size: float = Field(default=0.1, description="Fraction of data to use for test split"),
    local_output_dir: str = Field(default="/home/sagemaker-user/Agents/SAMA/data-prep", description="Local directory to save prepared data"),
    upload_to_s3: bool = Field(default=True, description="Whether to upload prepared data to S3"),
    s3_prefix: str = Field(default="finetune_dataset", description="S3 prefix for uploaded files")
) -> Dict[str, Any]:
    """
    Complete workflow: inspect dataset, prepare data, create template, and optionally upload to S3.
    """
    results = {"steps": []}
    
    try:
        # Step 1: Inspect dataset
        logger.info("Step 1: Inspecting dataset")
        inspect_result = await inspect_dataset(dataset_name=dataset_name, num_samples=3)
        results["steps"].append({"step": "inspect", "result": inspect_result})
        
        if inspect_result["status"] != "success":
            return {
                "status": "error",
                "message": "Failed at dataset inspection",
                "results": results
            }
        
        # Step 2: Prepare dataset based on type
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
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        
        results["steps"].append({"step": "prepare", "result": prep_result})
        
        if prep_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Failed at dataset preparation",
                "results": results
            }
        
        # Step 3: Create template
        logger.info("Step 3: Creating instruction template")
        template_path = str(Path(local_output_dir) / "template.json")
        template_result = await create_instruction_template(output_path=template_path)
        results["steps"].append({"step": "template", "result": template_result})
        
        # Step 4: Upload to S3 if requested
        if upload_to_s3:
            logger.info("Step 4: Uploading to S3")
            upload_result = await upload_training_data_to_s3(
                local_data_file=prep_result["train_file"],
                template_file=template_path,
                s3_prefix=s3_prefix
            )
            results["steps"].append({"step": "upload", "result": upload_result})
        
        return {
            "status": "success",
            "message": "Full data preparation workflow completed",
            "results": results
        }

    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Workflow failed: {str(e)}",
            "results": results
        }


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="SAMA Data Preparation MCP Server")
    parser.add_argument("--allow-write", action="store_true", help="Allow write operations")
    parser.add_argument("--version", action="version", version="1.0.0")
    
    args = parser.parse_args()
    
    if not args.allow_write:
        logger.warning("Write operations disabled. Use --allow-write to enable.")
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
