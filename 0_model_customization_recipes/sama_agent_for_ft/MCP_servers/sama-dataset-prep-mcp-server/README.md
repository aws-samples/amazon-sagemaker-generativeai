# SAMA Dataset Prep MCP Server

Search HF Hub for datasets, format them into messages-style JSONL for different modalities, and upload to S3.

## Tools

| Tool | Description |
|---|---|
| `search_datasets` | Search HuggingFace Hub for datasets by keyword |
| `inspect_dataset` | Preview schema, columns, and sample rows |
| `prepare_dataset` | Convert to messages JSONL (text, text_reasoning, image, audio) |
| `upload_to_s3` | Upload prepared dataset to S3 for training |

## Modality Formats

- **text**: Standard system/user/assistant messages
- **text_reasoning**: Extracts `<think>` blocks into `thinking` field (GPT-oss, DeepSeek-R1)
- **image**: Multimodal messages with `image_url` content blocks
- **audio**: Messages with audio content blocks

## Installation

```bash
pip install -e .
```
