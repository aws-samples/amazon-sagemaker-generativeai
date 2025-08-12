# ScribbleBots: Intelligent Research & Writing Agents with DeepSeek R1 and CrewAI

This project demonstrates how to build an intelligent multi-agent system using DeepSeek's R1 Distilled LLaMA model (70B parameters), CrewAI, and Amazon SageMaker. The system, called ScribbleBots, consists of specialized agents that work together to perform research and create high-quality written content.

## Architecture Overview

ScribbleBots uses a sequential workflow with two main agents:
- **Research Agent**: Gathers and analyzes information from various sources
- **Writer Agent**: Transforms research findings into polished, structured content

Both agents leverage the DeepSeek R1 Distilled LLaMA 70B model deployed on SageMaker for their cognitive capabilities.

## Prerequisites

- AWS Account with SageMaker access
- Hugging Face account and API token
- Python 3.8+
- Access to ml.p4d.24xlarge instance type on SageMaker
- S3 bucket for model artifacts

## Required Packages

```bash
pip install -qU crewai boto3 sagemaker streamlit==1.38.0 huggingface_hub psutil pynvml
pip install --upgrade transformers==4.44.2 torch>=1.1.13 torchvision torchaudio
```

## Environment Setup

1. Configure your environment variables:
```python
bucket_name = "your-bucket-name"
HUGGING_FACE_HUB_TOKEN = "your-huggingface-token"
my_region_name = "your-aws-region"
```

2. Deploy the DeepSeek model on SageMaker:
- Uses HuggingFace Deep Learning Container
- Requires 8 GPUs (ml.p4d.24xlarge instance)
- Model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B

## Project Structure

```
.
├── notebooks/
│   └── Blog1-CrewAI-DeekSeek.ipynb
├── tools/
│   └── sage_tools.py
├── images/
│   └── error.png
└── README.md
```

## Usage

1. First, run the setup cells to configure your environment and deploy the model.

2. Create the agents and tools:
```python
research_agent = Agent(
    role="Research Bot",
    goal="Scan sources, extract relevant information, and compile a research summary.",
    tools=[deepseek_tool],
    llm=DeepSeekSageMakerLLM(endpoint=custom_endpoint_name)
)

writer_agent = Agent(
    role="Writer Bot",
    goal="Transform research into structured content.",
    tools=[deepseek_tool],
    llm=DeepSeekSageMakerLLM(endpoint=custom_endpoint_name)
)
```

3. Execute the workflow:
```python
scribble_bots = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

result = scribble_bots.kickoff(inputs={"prompt": "Your research query here"})
```

## Cost Considerations

- The project uses ml.p4d.24xlarge instances which can be expensive
- Consider using smaller instances for development/testing
- Remember to delete endpoints when not in use

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Note

This is an experimental project and the model deployment requires significant computational resources. Make sure you have appropriate AWS quotas and budget before deployment.