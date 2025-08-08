# Generative AI using Amazon SageMaker

A comprehensive repository showcasing end-to-end Generative AI workflows on Amazon SageMaker, from getting started to production-ready implementations. This collection includes examples for model training, fine-tuning, inference, MLOps, distributed training, RAG systems, AI agents, and real-world use cases.

## 🚀 Quick Start

New to Generative AI on SageMaker? Start here:
- **[Getting Started Guide](1._getting_started/)** - Essential setup and foundational concepts

## 📚 Repository Structure

### 🎯 [End-to-End GenAI Lifecycle](2_end_to_end_genai_on_sagemaker/)
Complete workflows covering the entire ML lifecycle:
- **[Model Customization](2_end_to_end_genai_on_sagemaker/2_model_customization/)** - Fine-tuning and customization techniques
- **[Inference](2_end_to_end_genai_on_sagemaker/3_inference/)** - Deployment and serving strategies  
- **[MLOps](2_end_to_end_genai_on_sagemaker/4_mlops/)** - Production pipelines with SageMaker Pipelines for preprocessing, training, evaluation, and batch transform

### ⚡ [Distributed Training](3_distributed_training/)
Scalable training implementations for Large Language Models:
- **SageMaker Unified Studio** - Native distributed training capabilities
- **FSDP (Fully-Sharded Data Parallel)** - Hugging Face FSDP with SageMaker
- **Reinforcement Learning** - DPO and GRPO implementations with TRL, Unsloth, and veRL
- **Unsloth Fine-tuning** - Efficient instruction fine-tuning examples

### 🔍 [Retrieval-Augmented Generation (RAG)](4_rag/)
Knowledge-enhanced AI systems:
- **VoyageAI Embedding RAG** - Advanced RAG implementation with VoyageAI embeddings and Claude 3

### 🤖 [AI Agents](5_agents/)
Intelligent agent frameworks and implementations:
- **DeepSeek CrewAI Agent** - Multi-agent systems using CrewAI framework
- **LangGraph Model Context Protocol** - Advanced agent workflows with MCP integration
- **ML Models as Agent Tools** - Using ML models within agent architectures
- **SageMaker Strands Integration** - Enterprise-grade agent solutions

### 🎯 [Use Cases](6_use_cases/)
Real-world applications and industry solutions:
- **RAG & Chatbots** - Conversational AI with knowledge retrieval
- **Text Summarization** - Document and content summarization
- **Text Summarization to Image** - Multi-modal content generation
- **Text-to-SQL** - Natural language database querying

### 🚀 [Inference Optimization](7_inference/)
Performance and efficiency improvements:
- **Post-Training Quantization** - Model compression and optimization techniques

### 📊 [LLM Performance Evaluation](llm-performance-evaluation/)
Benchmarking and performance analysis:
- **DeepSeek R1 Distilled** - Performance evaluation and benchmarking tools

### 📦 [Archive](x_archive/)
Legacy examples and deprecated implementations for reference

## 🛠️ Key Features

- **Complete ML Lifecycle**: From data preprocessing to production deployment
- **Multiple Training Strategies**: Single-node, distributed, and reinforcement learning approaches  
- **Production-Ready MLOps**: Automated pipelines with SageMaker Pipelines
- **Advanced AI Patterns**: RAG, agents, and multi-modal applications
- **Performance Optimization**: Quantization, distributed training, and inference optimization
- **Industry Use Cases**: Real-world applications across various domains

## 🏗️ Technologies & Frameworks

- **Amazon SageMaker** - Core ML platform and services
- **Hugging Face Transformers** - Model implementations and fine-tuning
- **LangGraph & LangChain** - Agent frameworks and workflows
- **CrewAI** - Multi-agent system orchestration
- **Unsloth** - Efficient fine-tuning framework
- **TRL (Transformer Reinforcement Learning)** - RLHF implementations
- **FSDP & Ray** - Distributed training frameworks

## 📋 Prerequisites

- AWS Account with SageMaker access
- Python 3.8+ environment
- Basic understanding of machine learning concepts
- Familiarity with Jupyter notebooks

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd generative-ai-sagemaker
   ```

2. **Start with the basics**:
   - Review the [Getting Started Guide](1._getting_started/)
   - Explore [End-to-End Examples](2_end_to_end_genai_on_sagemaker/)

3. **Choose your path**:
   - **Beginners**: Start with basic inference and fine-tuning examples
   - **Intermediate**: Explore distributed training and RAG implementations  
   - **Advanced**: Dive into agents, MLOps pipelines, and custom use cases

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Reporting bugs and feature requests
- Submitting pull requests
- Code standards and best practices

## 🔒 Security

For security issue notifications, please see [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## 📄 License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Join community discussions for questions and ideas
- **Documentation**: Each directory contains detailed README files and examples

---

**Ready to build the future of AI?** Start exploring the examples and building your next Generative AI application on Amazon SageMaker! 🚀

