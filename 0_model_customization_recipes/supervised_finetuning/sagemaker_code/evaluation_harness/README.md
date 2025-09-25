# evaluation-harness

### Installation

1. `poetry install`
2. `poetry run evalharness --config evalconfig.yaml`

### Usage

Evaluation Harness is a declarative SDK that allows users to evaluate LLMs.

The user defines a yaml file with their specific config options:

```yaml
test_dataset: local_path.jsonl
# Need to have either source model or source model preds
source_model_path: local path
source_model_predictions_path: local_path.jsonl
# Need to have either destination model or destination model preds
target_model_path: local path
target_model_predictions_path: local_path.jsonl
# Reference model is optional
reference_model: sagemaker/custom-endpoint
reference_model_predictions_path: local_path.jsonl
# MLflow tracking server arn
mlflow_tracking_server_arn: arn
mlflow_experiment_name: experiment_name

metrics: ['bert', 'rouge2', 'toxicity', 'bleu', 'answer_similarity']
model_judge: bedrock:/us.anthropic.claude-3-5-haiku-20241022-v1:0
model_judge_parameters:
  temperature: 0
  max_tokens: 256
  anthropic_version: bedrock-2023-05-31
custom_metrics: ['??']
```

Once the yaml file is created, it is passed as a command-line argument to the Evaluation Harness SDK:

`python3 evalharness --config evalconfig.yaml`

Model prediction JSON Lines files should have the following format:

```jsonl
{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful AI...",
            "thinking": null
        },
        {
            "role": "user",
            "content": "Help me with...",
            "thinking": null
        },
        {
            "role": "assistant",
            "content": "Sure, your request...",
            "thinking": "It seems like..."
        }
    ],
    "ground_truth": [
        {
            "role": "assistant",
            "content": "You request...",
            "thinking": "The user asked..."
        }
    ],
}
```

The actual model predictions will be the assistant content in messages.

Ground truth will be the content field in `ground_truth`.

If you don't have prediction files precomputed, you'll need to pass in a test dataset.

The `test_dataset` option in the yaml file expects a JSON Lines file in the above format as well.
