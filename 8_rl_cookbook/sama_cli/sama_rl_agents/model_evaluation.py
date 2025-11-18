#!/usr/bin/env python3
"""
SAMA RL Model Evaluation MCP Server
Helps users evaluate deployed GRPO models with various metrics and techniques
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for sama_rl imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sama_rl"))

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
except ImportError:
    print("MCP not installed. Install with: pip install mcp")
    sys.exit(1)

# Check SAMA RL availability at startup
sama_rl = None  # Lazy load when needed

# Initialize server
server = Server("sama-rl-model-evaluation")

# Evaluation techniques
EVALUATION_TECHNIQUES = {
    "length_analysis": {
        "description": "Analyze response length distribution and target accuracy",
        "metrics": ["avg_length", "length_variance", "target_accuracy"],
        "use_cases": ["length-controlled generation", "summarization", "content creation"]
    },
    "sentiment_analysis": {
        "description": "Evaluate sentiment and tone of generated responses",
        "metrics": ["positive_ratio", "negative_ratio", "sentiment_score"],
        "use_cases": ["customer service", "content moderation", "brand voice"]
    },
    "quality_assessment": {
        "description": "Assess overall quality, coherence, and relevance",
        "metrics": ["coherence_score", "relevance_score", "fluency_score"],
        "use_cases": ["general text generation", "dialogue systems", "content quality"]
    },
    "task_specific": {
        "description": "Custom evaluation for specific tasks and domains",
        "metrics": ["task_accuracy", "domain_relevance", "custom_metrics"],
        "use_cases": ["domain-specific applications", "specialized tasks"]
    },
    "comparative_analysis": {
        "description": "Compare multiple models or versions",
        "metrics": ["relative_performance", "improvement_metrics", "regression_analysis"],
        "use_cases": ["model comparison", "A/B testing", "version evaluation"]
    }
}

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available tools for model evaluation"""
    return [
        types.Tool(
            name="list_evaluation_techniques",
            description="List all available evaluation techniques and their use cases",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="recommend_evaluation_strategy",
            description="Recommend evaluation strategy based on model type and use case",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "description": "Type of model (e.g., length-controlled, sentiment-aware)"
                    },
                    "use_case": {
                        "type": "string",
                        "description": "Intended use case for the model"
                    },
                    "evaluation_goals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific evaluation goals"
                    }
                },
                "required": ["model_type", "use_case"]
            }
        ),
        types.Tool(
            name="generate_evaluation_code",
            description="Generate evaluation code for a specific technique",
            inputSchema={
                "type": "object",
                "properties": {
                    "endpoint_name": {
                        "type": "string",
                        "description": "SageMaker endpoint name"
                    },
                    "evaluation_technique": {
                        "type": "string",
                        "enum": list(EVALUATION_TECHNIQUES.keys()),
                        "description": "Evaluation technique to use"
                    },
                    "test_dataset": {
                        "type": "string",
                        "description": "Test dataset or prompts to use"
                    },
                    "target_metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to calculate"
                    }
                },
                "required": ["endpoint_name", "evaluation_technique"]
            }
        ),
        types.Tool(
            name="create_evaluation_report",
            description="Create a comprehensive evaluation report template",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the model being evaluated"
                    },
                    "evaluation_techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of evaluation techniques to include"
                    },
                    "baseline_model": {
                        "type": "string",
                        "description": "Baseline model for comparison (optional)"
                    }
                },
                "required": ["model_name"]
            }
        ),
        types.Tool(
            name="setup_evaluation_environment",
            description="Generate code to set up evaluation environment with required dependencies",
            inputSchema={
                "type": "object",
                "properties": {
                    "evaluation_type": {
                        "type": "string",
                        "enum": ["basic", "advanced", "research"],
                        "description": "Type of evaluation setup needed"
                    },
                    "additional_libraries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional libraries needed"
                    }
                },
                "required": ["evaluation_type"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls"""
    
    if name == "list_evaluation_techniques":
        techniques_text = "**Available Evaluation Techniques:**\n\n"
        
        for technique, info in EVALUATION_TECHNIQUES.items():
            techniques_text += f"**{technique.replace('_', ' ').title()}**\n"
            techniques_text += f"- Description: {info['description']}\n"
            techniques_text += f"- Metrics: {', '.join(info['metrics'])}\n"
            techniques_text += f"- Use Cases: {', '.join(info['use_cases'])}\n\n"
        
        return [types.TextContent(type="text", text=techniques_text)]
    
    elif name == "recommend_evaluation_strategy":
        model_type = arguments.get("model_type")
        use_case = arguments.get("use_case")
        goals = arguments.get("evaluation_goals", [])
        
        # Recommendation logic based on model type and use case
        recommendations = []
        
        if "length" in model_type.lower() or "summarization" in use_case.lower():
            recommendations.append("length_analysis")
        
        if "sentiment" in model_type.lower() or "customer" in use_case.lower():
            recommendations.append("sentiment_analysis")
        
        if "quality" in goals or "coherence" in goals:
            recommendations.append("quality_assessment")
        
        if "compare" in goals or "baseline" in goals:
            recommendations.append("comparative_analysis")
        
        # Default to quality assessment if no specific recommendations
        if not recommendations:
            recommendations.append("quality_assessment")
        
        strategy_text = f"""
**Evaluation Strategy Recommendation**

**Model Type**: {model_type}
**Use Case**: {use_case}
**Goals**: {', '.join(goals) if goals else 'General evaluation'}

**Recommended Techniques**:
"""
        
        for technique in recommendations:
            info = EVALUATION_TECHNIQUES[technique]
            strategy_text += f"\n**{technique.replace('_', ' ').title()}**\n"
            strategy_text += f"- {info['description']}\n"
            strategy_text += f"- Key Metrics: {', '.join(info['metrics'])}\n"
        
        strategy_text += f"""

**Implementation Priority**:
1. Start with {recommendations[0].replace('_', ' ')} for immediate insights
2. Add {recommendations[1].replace('_', ' ') if len(recommendations) > 1 else 'quality assessment'} for comprehensive evaluation
3. Consider comparative analysis if you have baseline models

**Next Steps**:
- Use 'generate_evaluation_code' to get implementation code
- Set up evaluation environment with required dependencies
- Run evaluations on representative test data
"""
        
        return [types.TextContent(type="text", text=strategy_text)]
    
    elif name == "generate_evaluation_code":
        endpoint_name = arguments.get("endpoint_name")
        technique = arguments.get("evaluation_technique")
        test_dataset = arguments.get("test_dataset", "trl-lib/tldr")
        target_metrics = arguments.get("target_metrics", [])
        
        if technique == "length_analysis":
            code = f"""
# Length Analysis Evaluation
import boto3
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

# Setup
endpoint_name = "{endpoint_name}"
sagemaker_runtime = boto3.client('sagemaker-runtime')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)

# Load test data
dataset = load_dataset("{test_dataset}", split="test[:50]")
test_prompts = [item["prompt"] for item in dataset]

# Run inference and collect results
results = []
for prompt in test_prompts:
    # Call endpoint
    payload = {{"inputs": prompt, "parameters": {{"max_new_tokens": 200}}}}
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    completion = result[0]["generated_text"].replace(prompt, "").strip()
    
    # Calculate metrics
    tokens = tokenizer(completion, add_special_tokens=False)["input_ids"]
    token_count = len(tokens)
    
    results.append({{"prompt": prompt, "completion": completion, "token_count": token_count}})

# Analysis
token_counts = [r["token_count"] for r in results]
avg_length = np.mean(token_counts)
length_variance = np.var(token_counts)
target_length = 400  # Adjust based on your model's target

# Target accuracy (within 20% of target)
target_accuracy = sum(1 for tc in token_counts if abs(tc - target_length) <= target_length * 0.2) / len(token_counts)

print(f"Length Analysis Results:")
print(f"Average Length: {{avg_length:.1f}} tokens")
print(f"Length Variance: {{length_variance:.1f}}")
print(f"Target Accuracy: {{target_accuracy:.1%}}")
print(f"Length Range: {{min(token_counts)}} - {{max(token_counts)}} tokens")
"""
        
        elif technique == "sentiment_analysis":
            code = f"""
# Sentiment Analysis Evaluation
import boto3
import json
from datasets import load_dataset

# Setup
endpoint_name = "{endpoint_name}"
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Sentiment word lists
positive_words = ['good', 'great', 'excellent', 'amazing', 'helpful', 'useful', 'clear', 'wonderful']
negative_words = ['bad', 'terrible', 'awful', 'horrible', 'useless', 'wrong', 'confusing', 'poor']

# Load test data
dataset = load_dataset("{test_dataset}", split="test[:50]")
test_prompts = [item["prompt"] for item in dataset]

# Run evaluation
results = []
for prompt in test_prompts:
    # Call endpoint
    payload = {{"inputs": prompt, "parameters": {{"max_new_tokens": 200}}}}
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    completion = result[0]["generated_text"].replace(prompt, "").strip()
    
    # Sentiment analysis
    text_lower = completion.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    sentiment_score = positive_count - negative_count
    sentiment_label = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
    
    results.append({{
        "prompt": prompt,
        "completion": completion,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label
    }})

# Analysis
positive_ratio = sum(1 for r in results if r["sentiment_label"] == "positive") / len(results)
negative_ratio = sum(1 for r in results if r["sentiment_label"] == "negative") / len(results)
neutral_ratio = sum(1 for r in results if r["sentiment_label"] == "neutral") / len(results)
avg_sentiment_score = sum(r["sentiment_score"] for r in results) / len(results)

print(f"Sentiment Analysis Results:")
print(f"Positive Responses: {{positive_ratio:.1%}}")
print(f"Negative Responses: {{negative_ratio:.1%}}")
print(f"Neutral Responses: {{neutral_ratio:.1%}}")
print(f"Average Sentiment Score: {{avg_sentiment_score:.2f}}")
"""
        
        elif technique == "quality_assessment":
            code = f"""
# Quality Assessment Evaluation
import boto3
import json
from datasets import load_dataset

# Setup
endpoint_name = "{endpoint_name}"
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Quality indicators
quality_indicators = {{
    "coherence": ["clear", "logical", "consistent", "coherent"],
    "relevance": ["relevant", "appropriate", "related", "pertinent"],
    "fluency": ["smooth", "natural", "fluent", "well-written"]
}}

# Load test data
dataset = load_dataset("{test_dataset}", split="test[:50]")
test_prompts = [item["prompt"] for item in dataset]

# Run evaluation
results = []
for prompt in test_prompts:
    # Call endpoint
    payload = {{"inputs": prompt, "parameters": {{"max_new_tokens": 200}}}}
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    completion = result[0]["generated_text"].replace(prompt, "").strip()
    
    # Quality metrics
    text_lower = completion.lower()
    
    # Simple quality scoring
    coherence_score = sum(1 for word in quality_indicators["coherence"] if word in text_lower)
    relevance_score = sum(1 for word in quality_indicators["relevance"] if word in text_lower)
    fluency_score = len(completion.split()) / max(completion.count('.'), 1)  # Words per sentence
    
    # Overall quality (simple heuristic)
    overall_quality = (coherence_score + relevance_score + min(fluency_score/10, 1)) / 3
    
    results.append({{
        "prompt": prompt,
        "completion": completion,
        "coherence_score": coherence_score,
        "relevance_score": relevance_score,
        "fluency_score": fluency_score,
        "overall_quality": overall_quality
    }})

# Analysis
avg_coherence = sum(r["coherence_score"] for r in results) / len(results)
avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
avg_fluency = sum(r["fluency_score"] for r in results) / len(results)
avg_quality = sum(r["overall_quality"] for r in results) / len(results)

print(f"Quality Assessment Results:")
print(f"Average Coherence Score: {{avg_coherence:.2f}}")
print(f"Average Relevance Score: {{avg_relevance:.2f}}")
print(f"Average Fluency Score: {{avg_fluency:.2f}}")
print(f"Overall Quality Score: {{avg_quality:.2f}}")
"""
        
        else:
            code = f"# Evaluation technique '{technique}' not implemented yet"
        
        return [types.TextContent(
            type="text",
            text=f"Evaluation code for {technique}:\n\n```python{code}```"
        )]
    
    elif name == "create_evaluation_report":
        model_name = arguments.get("model_name")
        techniques = arguments.get("evaluation_techniques", ["length_analysis", "quality_assessment"])
        baseline = arguments.get("baseline_model")
        
        report_template = f"""
# Model Evaluation Report: {model_name}

## Executive Summary
This report provides a comprehensive evaluation of the {model_name} model using multiple assessment techniques.

## Model Information
- **Model Name**: {model_name}
- **Evaluation Date**: {{datetime.now().strftime('%Y-%m-%d')}}
- **Baseline Model**: {baseline or 'None'}
- **Evaluation Techniques**: {', '.join(techniques)}

## Evaluation Results

### 1. Performance Metrics
"""
        
        for technique in techniques:
            info = EVALUATION_TECHNIQUES.get(technique, {})
            report_template += f"""
#### {technique.replace('_', ' ').title()}
- **Description**: {info.get('description', 'Custom evaluation')}
- **Key Metrics**: {', '.join(info.get('metrics', ['Custom metrics']))}
- **Results**: [Insert results here]
"""
        
        report_template += """
### 2. Comparative Analysis
[Compare with baseline model if available]

### 3. Strengths and Weaknesses
**Strengths:**
- [List model strengths based on evaluation]

**Areas for Improvement:**
- [List areas needing improvement]

### 4. Recommendations
- [Provide actionable recommendations]

### 5. Conclusion
[Summarize overall model performance and suitability for intended use case]

## Appendix
### A. Evaluation Methodology
[Detailed description of evaluation methods]

### B. Raw Data
[Include raw evaluation data and statistics]
"""
        
        return [types.TextContent(
            type="text",
            text=f"Evaluation report template:\n\n```markdown{report_template}```"
        )]
    
    elif name == "setup_evaluation_environment":
        eval_type = arguments.get("evaluation_type")
        additional_libs = arguments.get("additional_libraries", [])
        
        base_requirements = [
            "boto3",
            "transformers",
            "datasets", 
            "numpy",
            "pandas",
            "matplotlib",
            "seaborn"
        ]
        
        if eval_type == "advanced":
            base_requirements.extend([
                "scikit-learn",
                "scipy",
                "nltk",
                "spacy"
            ])
        elif eval_type == "research":
            base_requirements.extend([
                "torch",
                "evaluate",
                "rouge-score",
                "bleu",
                "bertscore"
            ])
        
        all_requirements = base_requirements + additional_libs
        
        setup_code = f"""
# Evaluation Environment Setup

# Install required packages
import subprocess
import sys

packages = {all_requirements}

for package in packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"Installing {{package}}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Import required libraries
import boto3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from datasets import load_dataset

# Setup plotting
plt.style.use('default')
sns.set_palette("husl")

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')

print("Evaluation environment setup complete!")
print(f"Installed packages: {{', '.join(packages)}}")
"""
        
        return [types.TextContent(
            type="text",
            text=f"Evaluation environment setup ({eval_type}):\n\n```python{setup_code}```"
        )]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    """Main entry point"""
    # Check if SAMA RL is available
    try:
        import sama_rl
    except ImportError:
        print("ERROR: SAMA RL not installed. Please run: pip install -e . from the project root directory")
        sys.exit(1)
    
    # Run server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
