# Generative AI using Amazon SageMaker

A comprehensive repository showcasing production-ready Generative AI workflows on Amazon SageMaker. This collection provides end-to-end implementations spanning the complete ML lifecycle, from foundational concepts to enterprise-scale deployments, covering model training, fine-tuning, inference optimization, MLOps automation, distributed training, RAG systems, intelligent agents, and real-world industry applications.

## 🚀 Quick Start

New to Generative AI on SageMaker? Start here:

- **[Getting Started Guide](1._getting_started/)** - Essential setup, foundational concepts, and first steps

## 🤖 Models

This repository supports a comprehensive range of foundation models with various training methodologies. The table below shows model compatibility with different fine-tuning techniques, training frameworks, and deployment options.

### Model Support Matrix

| Models - Size                | Use Case / Strategy                             | Notebook                                                                                                                                                                                                                 | Service                        | Frameworks & Libs                                                              |
| ---------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------ |
| Qwen 3 0.6B                  | Function Calling, Agentic AI (FSDP, SFT, QLoRA) | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/fsdp/huggingface/model-trainer-notebook.ipynb)                                                                  | SageMaker AI Training Jobs     | Transformers, Accelerate, SageMaker Model Trainer, MLflow, Weights & Biases    |
| Qwen 3 0.6B                  | Function Calling, Agentic AI (LoRA, DPO)        | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/reinforcement-learning/dpo/trl/model-trainer-notebook.ipynb)                                                    | SageMaker AI Training Jobs     | Transformers, Accelerate, SageMaker Model Trainer, MLflow, Weights & Biases    |
| Arcee-Lite                   | Reasoning (FSDP, QLoRA, GRPO)                   | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/tree/main/3_distributed_training/reinforcement-learning/grpo/trl/torchrun/fsdp)                                                                  | SageMaker AI Training Jobs     | Transformers, Accelerate, SageMaker Model Trainer, MLflow, Weights & Biases    |
| Qwen 3 0.6B                  | Reasoning (FSDP, SFT, LoRA)                     | [Notebook](https://github.com/aws-samples/sample-ray-on-amazon-sagemaker-training-jobs/tree/main/examples/ray-torchtrainer/huggingface-grafana)                                                                          | SageMaker AI Training Jobs     | Ray, Grafana, Prometheus, Transformers, SageMaker Model Trainer                |
| Qwen 3 0.6B                  | Reasoning (FSDP, SFT, LoRA)                     | [Notebook](https://github.com/aws-samples/sample-ray-on-amazon-sagemaker-training-jobs/tree/main/examples/ray-torchtrainer/huggingface-heterogeneous-grafana)                                                            | SageMaker AI Training Jobs     | Heterogeneous Cluster, Ray, Grafana, Prometheus, SageMaker Estimator           |
| DeepSeek-R1-Distill-Llama-8B | Reasoning (SFT, QLoRA)                          | [Notebook](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/fine-tuning-with-sagemakerai-and-bedrock/task_02_customize_foundation_model/02.01_finetune_deepseekr1.ipynb)             | SageMaker AI Training Jobs     | Transformers, Accelerate, SageMaker Model Trainer, MLflow                      |
| GTE-Base-En-V1.5             | Embeddings                                      | [Notebook](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/blob/main/workshops/building-rag-workflows-with-sagemaker-and-bedrock/03-02_fine-tuning-embedding/01-ft_embedding_with_sagemaker_eval.ipynb) | SageMaker AI Training Jobs     | Sentence Transformers, Accelerate, SageMaker Estimator                         |
| Qwen 2 0.5B Instruct         | Summarization (GRPO)                            | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/reinforcement-learning/grpo/trl/accelerate/launch-training-job.ipynb)                                           | SageMaker AI Training Jobs     | Accelerate, Datasets, SageMaker, Transformers, TRL, Weights & Biases           |
| Gemma 3 4B-It                | Conversations, Reasoning (LoRA)                 | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/unsloth/instruct-fine-tuning-example-2/gemma3-4b-it.ipynb)                                                      | SageMaker AI Training Jobs     | Torch, TorchVision, TorchAudio, Unsloth, Psutil                                |
| Qwen 2 7B                    | Reasoning (GRPO)                                | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/reinforcement-learning/grpo/veRL/verl-on-sagemaker.ipynb)                                                       | SageMaker AI Training Jobs     | Verl, Torch, vLLM, FlashAttention                                              |
| Qwen 3 8B                    | Conversations (Spectrum)                        | [Notebook](https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/21_spectrum_finetuning/spectrum_training.ipynb)                                                                              | SageMaker AI Training Jobs     | Transformers, Accelerate, SageMaker Model Trainer, MLflow, Weights & Biases    |
| Meta LLaMA 3.2 3B            | Function Calling, Agentic AI (SFT, LoRA, DPO)   | [Notebook](https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/22_dpo_alignment_trl_sagemaker/run_training_job.ipynb)                                                                       | SageMaker AI Training Jobs     | Accelerate, Datasets, SageMaker, Transformers, TRL, Weights & Biases           |
| Qwen 2.5 0.5B Instruct       | Reasoning (GRPO)                                | [Notebook](https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/20_grpo_trl_sagemaker/grpo-test.ipynb)                                                                                       | SageMaker AI Training Jobs     | Accelerate, Datasets, SageMaker, Transformers, TRL                             |
| LLaMA 3 8B Instruct          | Reasoning, Conversation (SFT, LoRA, QLoRA, KD)  | [Notebook](https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/19_knowledge_distillation/test_gkd_deepseek.ipynb)                                                                           | SageMaker AI Training Jobs     | Accelerate, Datasets, SageMaker, Transformers, TRL, TorchRun, Weights & Biases |
| LLaMA 3 / LLaMA 2 / Mistral  | Text Generation (FSDP)                          | [Notebook](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/pytorch/FSDP)                                                                                                               | SageMaker HyperPod (Slurm/EKS) | PyTorch, SMHP Training Operator                                                |
| GPT on NeMo                  | Text Generation (Spectrum)                      | [Notebook](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/megatron/nemo)                                                                                                              | SageMaker HyperPod (Slurm/EKS) | NVIDIA NeMo                                                                    |
| SMoLM 1.7B on Picotron       | Text Generation (FSDP)                          | [Notebook](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/pytorch/picotron)                                                                                                           | SageMaker HyperPod (Slurm/EKS) | Hugging Face Picotron                                                          |
| LLaMA 3.1 on TorchTitan      | Text Generation (FSDP, Spectrum)                | [Notebook](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/pytorch/torchtitan)                                                                                                         | SageMaker HyperPod (Slurm/EKS) | PyTorch, TorchTitan                                                            |
| Qwen 2.5 72B w/ HF TRL       | Preference Alignment, Reasoning (GRPO)          | [Notebook](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/pytorch/trl/grpo)                                                                                                           | SageMaker HyperPod (Slurm/EKS) | PyTorch, Hugging Face TRL                                                      |
| Qwen 2.5 VL                  | Multimodality (SFT, QLoRA)                      | [Notebook](https://github.com/aws-samples/multi-modal-examples-for-amazon-sagemaker/blob/main/01-video_content_reel_generator-qwen2_vl/04-02_optional_fine_tune_video_inference.ipynb)                                   | SageMaker Training Jobs        | SWIFT                                                                          |
| Meta LLaMA 3 8B RLHF         | Preference Alignment (FSDP, DPO, QLoRA)         | [Notebook](https://github.com/aws-samples/sagemaker-studio-foundation-models/blob/main/use-cases/dpo/RLHF-with-Llama3-on-Studio-DPO.ipynb)                                                                               | SageMaker Training Jobs        | Hugging Face TRL                                                               |
| GPT-OSS 20B                  | Reasoning (Accelerate, SFT, MXFP4, vLLM)        | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/models/openai--gpt-oss/finetune_gpt_oss.ipynb)                                                                  | SageMaker Training Jobs        | Hugging Face Trainer, MXFP4                                                    |
| GPT-OSS 20B                  | Reasoning (SMDDP, SFT, MXFP4)                   | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/models/openai--gpt-oss/finetune_gpt_oss_hyperpod_recipes_eks.ipynb)                                             | SageMaker HyperPods (EKS)      | HyperPod Recipes                                                               |
| GPT-OSS 20B                  | Reasoning (SMDDP, SFT, MXFP4)                   | [Notebook](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/3_distributed_training/models/openai--gpt-oss/finetune_gpt_oss_hyperpod_recipes_tj.ipynb)                                              | SageMaker TrainingJobs         | HyperPod Recipes                                                               |
| LLaMA 3.1 8B Instruct        | Reasoning (FSDP, SFT, QLoRA)                    | [Notebook](https://github.com/brunopistone/amazon-sagemaker-generativeai/blob/main/3_distributed_training/models/meta-llama-3.1-8b/sft_llama_31_8b.ipynb)                                                                | SageMaker TrainingJobs         | Transformers, TRL, BitsAndBytes, Accelerate, MLflow, PEFT                      |

### Training Infrastructure

- **SageMaker Hyperpod** - High-performance computing clusters for large-scale training
- **SageMaker Training Jobs** - Standard managed training infrastructure

## 📚 Repository Structure

### 🎯 [End-to-End GenAI Lifecycle](2_end_to_end_genai_on_sagemaker/)

**Complete production workflows covering the entire ML lifecycle with enterprise-grade practices**

- **[Model Customization](2_end_to_end_genai_on_sagemaker/2_model_customization/)** - Advanced fine-tuning techniques including instruction tuning, parameter-efficient methods (LoRA, QLoRA), and domain adaptation strategies
- **[Inference](2_end_to_end_genai_on_sagemaker/3_inference/)** - Production deployment patterns, real-time and batch inference, auto-scaling configurations, and multi-model endpoints
- **[MLOps](2_end_to_end_genai_on_sagemaker/4_mlops/)** - Automated CI/CD pipelines using SageMaker Pipelines with integrated preprocessing, training, evaluation, model registration, and batch transform operations

### ⚡ [Distributed Training](3_distributed_training/)

**Scalable training implementations for Large Language Models with advanced parallelization strategies**

- **SageMaker Unified Studio** - Native distributed training capabilities with seamless cluster management and resource optimization
- **FSDP (Fully Sharded Data Parallel)** - Memory-efficient training using Hugging Face FSDP integration for models exceeding single-GPU memory limits
- **Reinforcement Learning from Human Feedback** - DPO (Direct Preference Optimization) and GRPO implementations using TRL, Unsloth, and veRL frameworks
- **Efficient Fine-tuning** - Unsloth-powered instruction fine-tuning with 2x-5x speed improvements and reduced memory consumption

### 🔍 [Retrieval-Augmented Generation (RAG)](4_rag/)

**Knowledge-enhanced AI systems with advanced embedding and retrieval techniques**

- **VoyageAI Embedding RAG** - Production-ready RAG implementation featuring VoyageAI's state-of-the-art embeddings, Claude 3 integration, vector database optimization, and semantic search capabilities for enterprise knowledge bases

### 🤖 [AI Agents](5_agents/)

**Intelligent multi-agent frameworks and orchestration systems**

- **[DeepSeek CrewAI Agent](5_agents/deepseek_crewai_based_agent/)** - Multi-agent research and writing system using DeepSeek R1 Distilled LLaMA 70B with CrewAI orchestration for collaborative task execution
- **[LangGraph Model Context Protocol](5_agents/langgraph_model_context_protocol/)** - Advanced agentic workflows with MCP integration for loan underwriting, featuring multi-step orchestration and role-based agent specialization
- **[ML Models as Agent Tools](5_agents/ml-models-as-agent-tools/)** - Integration patterns for using SageMaker-deployed ML models as agent tools via MCP, including both direct implementation and Amazon Bedrock AgentCore approaches
- **[SageMaker Strands Integration](5_agents/sagemaker-strands-agentcore/)** - Enterprise-grade agent solutions with managed hosting and authentication

### 🎯 [Use Cases](6_use_cases/)

**Real-world applications and industry-specific solutions**

- **[RAG & Chatbots](6_use_cases/usecases/rag_including_chatbot/)** - Conversational AI with knowledge retrieval using FLAN-T5-XL and Falcon-7B models, featuring document processing and context-aware responses
- **[Text Summarization](6_use_cases/usecases/text_summarization/)** - Document and content summarization using AI21, Falcon-7B, and FLAN-T5-XL models with LangChain integration
- **[Text Summarization to Image](6_use_cases/usecases/text_summarization_to_image/)** - Multi-modal content generation pipeline combining text summarization with image generation capabilities
- **[Text-to-SQL](6_use_cases/usecases/text_to_sql/)** - Natural language database querying using Code Llama with LangChain SQL query generation, complete with demo database and web interface

### 🚀 [Inference Optimization](7_inference/)

**Performance and efficiency improvements for production deployments**

- **[Post-Training Quantization](7_inference/post_training_quantization/)** - Model compression techniques using GPTQ and AWQ quantization methods, reducing memory footprint by 50-75% while maintaining accuracy, with automated SageMaker Training Job implementation

### 📊 [LLM Performance Evaluation](llm-performance-evaluation/)

**Comprehensive benchmarking and performance analysis frameworks**

- **[DeepSeek R1 Distilled](llm-performance-evaluation/deepseek-r1-distilled/)** - Performance evaluation and benchmarking tools for the DeepSeek R1 Distilled model series, including accuracy metrics, latency analysis, and cost optimization studies

### 📦 [Archive](x_archive/)

**Legacy examples and deprecated implementations for reference and migration guidance**

## 🛠️ Key Features

- **Complete ML Lifecycle**: From data preprocessing and model training to production deployment and monitoring
- **Multiple Training Strategies**: Single-node, multi-node distributed, and reinforcement learning approaches with automatic scaling
- **Production-Ready MLOps**: Automated CI/CD pipelines with SageMaker Pipelines, model registry, and deployment automation
- **Advanced AI Patterns**: RAG systems, multi-agent orchestration, and multi-modal applications
- **Performance Optimization**: Model quantization, distributed training, inference acceleration, and cost optimization
- **Industry Use Cases**: Financial services, healthcare, retail, and manufacturing applications
- **Enterprise Security**: IAM integration, VPC support, encryption at rest and in transit

## 🏗️ Technologies & Frameworks

### Core Platform

- **Amazon SageMaker** - Managed ML platform with training, inference, and MLOps capabilities
- **Amazon Bedrock** - Managed foundation model service with enterprise security
- **AWS Lambda & API Gateway** - Serverless inference and API management

### ML Frameworks

- **Hugging Face Transformers** - State-of-the-art model implementations and fine-tuning utilities
- **PyTorch & TensorFlow** - Deep learning frameworks with distributed training support
- **FSDP & DeepSpeed** - Memory-efficient distributed training frameworks
- **Ray** - Distributed computing framework for ML workloads

### Agent & Orchestration

- **LangGraph & LangChain** - Agent frameworks and workflow orchestration
- **CrewAI** - Multi-agent system coordination and task delegation
- **Model Context Protocol (MCP)** - Standardized tool integration for AI agents

### Optimization & Efficiency

- **Unsloth** - 2x-5x faster fine-tuning with reduced memory usage
- **TRL (Transformer Reinforcement Learning)** - RLHF and preference optimization
- **llm-compressor** - Post-training quantization with GPTQ and AWQ
- **vLLM** - High-throughput inference serving

## 📋 Prerequisites

### AWS Requirements

- AWS Account with SageMaker access and appropriate service limits
- IAM roles with SageMaker, S3, and related service permissions
- VPC configuration for secure deployments (optional but recommended)

### Development Environment

- Python 3.8+ with virtual environment management
- Jupyter Lab/Notebook for interactive development
- AWS CLI configured with appropriate credentials
- Git for version control and collaboration

### Knowledge Prerequisites

- Intermediate understanding of machine learning concepts
- Familiarity with Python programming and data science libraries
- Basic knowledge of AWS services and cloud computing
- Understanding of transformer architectures and LLMs (recommended)

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd generative-ai-sagemaker

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. AWS Configuration

```bash
# Configure AWS credentials
aws configure

# Verify SageMaker access
aws sagemaker list-training-jobs --max-items 1
```

### 3. Choose Your Learning Path

#### **Beginners** (New to GenAI/SageMaker)

1. Start with [Getting Started Guide](1._getting_started/)
2. Explore basic [Inference examples](2_end_to_end_genai_on_sagemaker/3_inference/)
3. Try simple [Use Cases](6_use_cases/) like text summarization

#### **Intermediate** (Some ML/Cloud experience)

1. Dive into [Model Customization](2_end_to_end_genai_on_sagemaker/2_model_customization/)
2. Explore [Distributed Training](3_distributed_training/) techniques
3. Implement [RAG systems](4_rag/) for knowledge-enhanced applications

#### **Advanced** (Production-ready implementations)

1. Master [MLOps pipelines](2_end_to_end_genai_on_sagemaker/4_mlops/)
2. Build [Multi-agent systems](5_agents/)
3. Optimize with [Quantization techniques](7_inference/post_training_quantization/)

### 4. Quick Validation

Run a simple inference example to validate your setup:

```python
# Example: Deploy a pre-trained model for text generation
from sagemaker.huggingface import HuggingFaceModel

model = HuggingFaceModel(
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py310",
    role=role,
    model_data="s3://path-to-model"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge"
)
```

## 🎯 Example Workflows

### Text Generation Pipeline

```
Data Preparation → Model Fine-tuning → Evaluation → Deployment → Monitoring
     ↓                    ↓              ↓           ↓           ↓
  S3 Storage      SageMaker Training   Model Registry  Endpoint   CloudWatch
```

### RAG Implementation

```
Document Ingestion → Embedding Generation → Vector Storage → Query Processing → Response Generation
        ↓                     ↓                 ↓              ↓                    ↓
   Text Processing      SageMaker Endpoint   Vector DB    Retrieval Logic    LLM Inference
```

### Multi-Agent System

```
Task Definition → Agent Orchestration → Tool Execution → Result Aggregation → Final Output
      ↓                  ↓                   ↓               ↓                  ↓
  LangGraph         CrewAI Framework    MCP Servers    Agent Coordination   Structured Response
```

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- **Reporting Issues**: Bug reports, feature requests, and documentation improvements
- **Code Contributions**: Pull requests, code reviews, and testing procedures
- **Standards**: Code formatting, documentation requirements, and best practices
- **Community Guidelines**: Code of conduct and collaboration expectations

### Contribution Areas

- New use case implementations
- Performance optimizations
- Documentation improvements
- Testing and validation
- Integration with new AWS services

## 🔒 Security

Security is our top priority. For security issue notifications and responsible disclosure, please see [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

### Security Best Practices

- Use IAM roles with least privilege access
- Enable encryption at rest and in transit
- Implement VPC endpoints for secure communication
- Regular security audits and compliance checks

## 📄 License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

## 🆘 Support & Resources

### Community Support

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Community Q&A and knowledge sharing
- **Documentation**: Comprehensive guides and API references

### AWS Resources

- **SageMaker Documentation**: [Official AWS Documentation](https://docs.aws.amazon.com/sagemaker/)
- **AWS Support**: Professional support plans available
- **AWS Training**: Certification and learning paths

### Additional Resources

- **Model Hub**: Pre-trained models and configurations
- **Best Practices**: Performance optimization and cost management
- **Case Studies**: Real-world implementation examples

---

## 🌟 What's New

- **Latest Updates**: DeepSeek R1 integration, enhanced MCP support, improved quantization techniques
- **Coming Soon**: Multi-modal agents, advanced RAG patterns, cost optimization tools
- **Community Highlights**: Featured implementations and success stories

---

**Ready to build the future of AI?** Start exploring the examples and building your next Generative AI application on Amazon SageMaker! 🚀

_This repository is actively maintained and regularly updated with the latest AWS services, model architectures, and best practices. Star ⭐ the repository to stay updated with new releases and features._
