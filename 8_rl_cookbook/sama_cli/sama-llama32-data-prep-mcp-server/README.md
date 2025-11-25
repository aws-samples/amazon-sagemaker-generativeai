# SAMA Data Preparation MCP Server

A Q CLI compatible MCP server for dataset preparation and formatting, specifically designed for SAMA (SageMaker AI Model Agents) context.

## Overview

This MCP server provides comprehensive data preparation capabilities for SAMA agents, enabling natural language interactions for dataset loading, formatting, and preparation for fine-tuning workflows.

## Features

This MCP server provides the following tools for SAMA agent data preparation operations:

### `prepare_instruction_dataset`
Prepare instruction-following datasets (e.g., Dolly):
- Load datasets from HuggingFace Hub
- Optional category filtering (e.g., 'summarization', 'closed_qa')
- Train/test splitting
- JSONL output format for training

### `prepare_summarization_dataset`
Prepare summarization datasets:
- Auto-detect text and summary fields
- Convert to instruction format
- Configurable field mapping
- Train/test splitting

### `prepare_qa_dataset`
Prepare Question-Answering datasets:
- Support for context-based and open QA
- Flexible answer format handling (string, dict, list)
- Auto-detect common field names
- Instruction format conversion

### `create_instruction_template`
Create prompt templates for fine-tuning:
- SageMaker JumpStart compatible format
- Configurable field placeholders
- JSON template output

### `upload_training_data_to_s3`
Upload prepared data to S3:
- Automatic SageMaker bucket detection
- Batch file upload (data + template)
- Configurable S3 prefixes

### `inspect_dataset`
Dataset inspection and preview:
- Column information and data types
- Sample record preview
- Dataset statistics

### `full_data_prep_workflow`
Complete end-to-end workflow:
- Dataset inspection
- Data preparation based on type
- Template creation
- Optional S3 upload
- Comprehensive result tracking

## Installation

```bash
cd /path/to/sama-data-prep-mcp-server
pip install -e .
```

## Configuration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "sama-data-prep-mcp-server": {
      "command": "sama-data-prep-mcp-server",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

## Prerequisites

- AWS credentials configured
- SageMaker permissions for S3 access
- HuggingFace datasets library
- Internet access for dataset downloads

## Usage Examples

### Dataset Inspection
- "Inspect the databricks/databricks-dolly-15k dataset"
- "Show me sample records from the CNN/DailyMail dataset"

### Data Preparation
- "Prepare databricks/databricks-dolly-15k for instruction tuning"
- "Prepare CNN/DailyMail dataset for summarization with test split 0.2"
- "Prepare SQuAD dataset for QA fine-tuning"

### Category Filtering
- "Prepare Dolly dataset filtering for summarization category only"
- "Load Dolly dataset with closed_qa category filter"

### Full Workflow
- "Do a full data prep iteration on databricks/databricks-dolly-15k"
- "Complete workflow for CNN/DailyMail summarization dataset with S3 upload"

### Template and Upload
- "Create instruction template with custom field names"
- "Upload training data to S3 with custom prefix"

## Supported Dataset Types

### Instruction-Following Datasets
- databricks/databricks-dolly-15k
- Anthropic/hh-rlhf
- OpenAssistant/oasst1
- And other instruction-tuning datasets

### Summarization Datasets
- cnn_dailymail
- xsum
- samsum
- And other text summarization datasets

### QA Datasets
- squad
- squad_v2
- natural_questions
- And other question-answering datasets

## Data Format

The server converts all datasets to a consistent instruction format:

```json
{
  "instruction": "The task instruction or question",
  "context": "Additional context or input text",
  "response": "The expected response or answer"
}
```

## Template Format

Generated templates follow SageMaker JumpStart format:

```json
{
  "prompt": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n",
  "completion": " {response}"
}
```

## Error Handling

The server includes comprehensive error handling for:
- Dataset loading failures
- Field mapping issues
- S3 upload problems
- AWS credential issues
- Network connectivity problems

## Integration with SAMA Workflow

This server is designed to work seamlessly with other SAMA MCP servers:
- **SAMA Deployment Server**: For model deployment after fine-tuning
- **SAMA Fine-tuning Server**: For training jobs (future)
- **SAMA Evaluation Server**: For model evaluation (future)

## Full Iteration Support

When you request a "full iteration" on a dataset, the server can coordinate with other SAMA servers to:
1. Prepare the dataset (this server)
2. Start fine-tuning job (fine-tuning server)
3. Deploy the fine-tuned model (deployment server)
4. Run evaluation (evaluation server)

Example: "Let's do a full iteration on databricks/databricks-dolly-15k"
