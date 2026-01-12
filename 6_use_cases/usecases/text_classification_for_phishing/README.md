# 🔒 Phishing Detection with Sequence Classification

Fine-tune Qwen2.5-1.5B for binary phishing detection using sequence classification on Amazon SageMaker AI.

## Why Sequence Classification for Security?

### 1. **Fast Inference** ⚡
Sequence classification generates a single token (the class label) rather than full text, resulting in:
- **Sub-second latency**: Single forward pass through the model
- **Predictable performance**: No variable-length generation
- **High throughput**: Can process thousands of requests per second

**Comparison**:
- Text generation (50 tokens): ~500-1000ms
- Sequence classification: ~50-150ms (5-10x faster)

### 2. **Low Cost** 💰
Small language models (SLMs) on small instances vs. large LLMs:
- **Model size**: 1.5B parameters vs. 70B+ parameters
- **Instance type**: `ml.g5.xlarge` (\$1.41/hr) vs. `ml.p4de.24xlarge` (\$31.56/hr)
- **Memory footprint**: ~3GB vs. ~140GB VRAM
- **Cost savings**: 20-50x reduction in infrastructure costs

**Cost Comparison for 1M email scans/day:**

| Method | Daily Cost | Monthly Cost | Cost per 1K scans |
|--------|-----------|--------------|-------------------|
| **Sequence classification (SageMaker AI)** | ~\$34 | ~\$1,020 | **\$0.034** |
| Text generation endpoint (SageMaker AI) | ~\$750+ | ~\$22,500+ | \$0.75+ |
| **Claude Haiku 4.5 (Bedrock)** | ~\$250 | ~\$7,500 | **\$0.25** |

- **For high-volume use cases (>10K/day)**, self-hosted models win dramatically
- **For low-volume (<5K/day)**, Bedrock models may be more cost-effective

*Note*: The cost figures above are based on internal benchmarking and testing under representative workloads. Actual execution costs may vary depending on factors such as model configuration, input size, traffic patterns, regional pricing, and infrastructure utilization. These numbers should be treated as indicative estimates rather than guaranteed or fixed pricing.

### 3. **No Data Leakage** 🔐
Critical for security and compliance:
- **Cannot regurgitate training data**: The model can only outputs class indices (0 or 1)
- **No prompt injection risk**: Input goes through classifier head, not generative decoder
- **Deterministic outputs**: Argmax over logits, no sampling
- **No memorization exposure**: Cannot reproduce sensitive emails from training set

**Security benefit**: Even in the unlikely case in which an attacker gains access to the endpoint, they cannot extract training data or inject malicious prompts to manipulate text generation.

### 4. **Own Your Model and IP** 🏢
- **Full control**: Train on your proprietary data
- **Customizable**: Fine-tune decision boundaries for your threat landscape
- **No vendor lock-in**: Deploy anywhere (SageMaker, on-prem, edge)
- **Regulatory compliance**: Keep sensitive data in-house (GDPR, HIPAA, etc.)
- **Version control**: Track model iterations and A/B test

### 5. **Quantifiable Performance** 📊
Security teams need precise metrics:
- **Precision**: How many flagged emails are actually malicious?
- **Recall**: What percentage of malicious emails are caught?
- **False Positive Rate**: How often do we block legitimate emails?
- **False Negative Rate**: How often do threats slip through?
- **F1 Score**: Balanced measure of precision and recall

Unlike LLMs that generate explanations, sequence classifiers provide:
- **Confidence scores**: Probability distribution over classes
- **Explainable predictions**: Logits show model certainty
- **Tunable thresholds**: Adjust sensitivity vs. specificity

## Technical Architecture

### Model: Qwen2.5-1.5B
- **Architecture**: Transformer-based decoder
- **Task**: Binary sequence classification (Safe vs. Phishing)
- **Fine-tuning**: RSLoRA (rank-stabilized LoRA) on classification head
- **Precision**: bfloat16 mixed precision
- **Training**: ~60-75 minutes on ml.g5.xlarge

### Dataset: `drorrabin/phishing_emails-data`
- **Size**: ~27k training samples, ~3.7k test samples
- **Format**: Email content with binary labels
- **Balance**: 50/50 safe vs. phishing in training set
- **Source**: [HuggingFace](https://huggingface.co/datasets/drorrabin/phishing_emails-data)

### Deployment: SageMaker + vLLM
- **Container**: LMI v18 with vLLM 0.12.0
- **Inference**: Text classification mode (single token prediction)
- **Instance**: `ml.g5.xlarge` (1x NVIDIA A10G, 24GB VRAM)
- **Routing**: Least Outstanding Requests for load balancing

## Repository Structure

```
phishing-detection-notebooks/
├── 01_data_processing.ipynb      # Load, preprocess, upload to S3
├── 02_model_training.ipynb       # Fine-tune with SageMaker + MLflow
├── 03_model_deployment.ipynb     # Deploy endpoint with vLLM
├── 04_benchmarking.ipynb         # Latency/throughput testing
├── utils.py                      # Helper functions (S3, model extraction)
└── README.md                     # This file
```

### Notebook Workflow

The notebooks are designed to run sequentially, with state passed via IPython's `%store` magic:

1. **01_data_processing.ipynb** → Stores: `train_s3_uri`, `val_s3_uri`, `test_s3_uri`, `NUM_LABELS`
2. **02_model_training.ipynb** → Stores: `model_s3_uri`, `training_job_name`, `mlflow_experiment_name`
3. **03_model_deployment.ipynb** → Stores: `endpoint_name`, `model_name`
4. **04_benchmarking.ipynb** → Uses stored endpoint info for testing

Each notebook:
- **Validates prerequisites** at the start
- **Documents expected outputs** in the header
- **Points to the next notebook** in the summary

## Prerequisites

### AWS Resources
- **SageMaker AI**: JupyterLab space or local environment with SageMaker SDK
- **IAM Role**: SageMaker execution role with S3 and SageMaker permissions
- **S3 Bucket**: For storing datasets and model artifacts
- **MLflow**: SageMaker MLflow app for experiment tracking

### Software Requirements
- Python 3.10+
- SageMaker SDK 2.253.1+
- Transformers, PEFT, datasets (installed in training container)

### Budget
- **Data Processing**: ~\$0.01 (S3 storage)
- **Model Training**: ~\$1.50-\$2.00 (60-75 min on `ml.g5.xlarge`)
- **Model Deployment**: ~\$1.41/hour (`ml.g5.xlarge` endpoint)
- **Total for full workflow**: ~\$4-6

*Note*: The cost figures above are based on internal benchmarking and testing under representative workloads. Actual execution costs may vary depending on factors such as model configuration, input size, traffic patterns, regional pricing, and infrastructure utilization. These numbers should be treated as indicative estimates rather than guaranteed or fixed pricing.

## Use Cases

### Email Security Gateway
- **Real-time**: Scan incoming emails before delivery
- **Throughput**: Thousands of emails per minute
- **Latency**: Sub-second classification required
- **Cost**: Fraction of API-based solutions

### SMS/Messaging Platforms
- **Mobile apps**: Protect users from phishing SMS
- **Chat platforms**: Flag malicious links in messages
- **Social media**: Content moderation at scale

### URL Reputation Systems
- **Web browsers**: Real-time URL scanning
- **Proxy servers**: Network-level protection
- **Security appliances**: Integrate with existing infrastructure

### Document Classification
- **Email attachments**: Scan PDFs, docs for malicious content
- **File uploads**: Validate user-submitted documents
- **Data loss prevention**: Identify sensitive information

## Advanced Topics

### Fine-tuning Your Own Model

1. **Collect your data**: Label emails as safe/phishing
2. **Format as JSONL**: `{"text": "...", "label": 0 or 1}`
3. **Upload to S3**: Replace dataset path in `01_data_processing.ipynb`
4. **Adjust hyperparameters**: Tune learning rate, LoRA rank, batch size
5. **Monitor MLflow**: Watch for overfitting

### Improving Model Performance

**For higher precision (fewer false positives)**:
- Increase classification threshold: Use `logits[1] > 0.8` instead of argmax
- Add hard negative mining: Include challenging safe examples
- Increase model capacity: Use Qwen2.5-3B or Qwen2.5-7B

**For higher recall (catch more threats)**:
- Lower classification threshold: Use `logits[1] > 0.3`
- Augment with adversarial examples: Add obfuscated phishing emails
- Ensemble models: Combine multiple checkpoints

**For multi-class classification**:
- Extend labels: Safe, Phishing, Spam, Malware, Social Engineering
- Modify `NUM_LABELS` in data processing
- Update training script `num_labels` parameter

## Troubleshooting

### Training Issues

**CUDA OOM errors during training**:
- Reduce `train_batch_size` (try 4 or 2)
- Increase `gradient_accumulation_steps` to compensate
- Use `fp16` instead of `bf16` (requires V100/A100 GPUs)

**Training too slow**:
- Increase `train_batch_size` (try 16 or 32)
- Use larger instance (ml.g5.2xlarge, ml.g5.4xlarge)
- Enable `fp16` mixed precision

**Poor validation metrics**:
- Train for more epochs (try 3-5)
- Increase LoRA rank (try 32 or 64)
- Check for data imbalance
- Increase learning rate (try 2e-4 or 5e-4)

## References

### Papers
- [RSLoRA: Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2309.16609)

### Documentation
- [SageMaker ModelTrainer](https://sagemaker.readthedocs.io/en/stable/api/training/model_trainer.html)
- [Managed MLflow on Amazon SageMaker AI](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow.html)
- [LMI Containers](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html)
- [vLLM Text Classification](https://docs.vllm.ai/en/v0.7.0/getting_started/examples/classification.html)

### Datasets
- [drorrabin/phishing_emails-data](https://huggingface.co/datasets/drorrabin/phishing_emails-data)
- [Original Kaggle Dataset](https://www.kaggle.com/datasets/subhajournal/phishingemails)