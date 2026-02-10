# GRPO Training with veRL on Amazon SageMaker

This folder contains two examples for running GRPO (Generalized Reinforcement Policy Optimization) training using the [veRL](https://github.com/volcengine/verl) framework on Amazon SageMaker.

## Examples

| Example                                                                          | Description                                                                                                                                                                                    |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [single-node](single-node/)                                                      | Generic veRL setup on SageMaker. Uses veRL's built-in trainer scripts with Ray running implicitly under the hood.                                                                              |
| [multi-node](../../../3_distributed_training/models/deepseek-r1-distill-qwen-7b) | Model-specific example for **DeepSeek-R1-Distill-Qwen-7B** with explicit Ray cluster orchestration across multiple SageMaker nodes. Includes custom training scripts, Dockerfile, and dataset. |
