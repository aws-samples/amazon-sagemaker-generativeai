# LLM Customization with Job Governance on Amazon SageMaker AI and AWS Batch

AWS Batch enables efficient queuing and resource management for your SageMaker Training Jobs.

## Getting Started

The instructions below are designed to get you going with this feature quickly.

### Prerequisites

1. **SageMaker Execution Role** — A role with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess` policies attached. This is the role passed to SageMaker for training execution. See [How to use SageMaker AI execution roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
2. **AWS Batch permissions** — The role running the notebooks needs permissions to call AWS Batch APIs (`batch:*`). If your role doesn't already have these, add them.
3. **Service-Linked Role** — The `AWSServiceRoleForAWSBatchWithSagemaker` role is created automatically when you create your first service environment. No manual setup required.

For the full setup guide, see [Getting started with AWS Batch for SageMaker Training](https://docs.aws.amazon.com/batch/latest/userguide/getting-started-sagemaker.html).

### Python setup

In order to use the feature, the python `boto3` library needs to be installed.

```
pip install -U boto3 pyyaml sagemaker>=3.7
```

### Create AWS Batch queues

Define your queues in [`smtj_batch_utils/config.yaml`](./smtj_batch_utils/config.yaml). The config supports three queue types: **FIFO**, **fair-share**, and **quota-management** (with lending, borrowing, and preemption). Then create the resources:

```bash
cd smtj_batch_utils && python3 create_resources.py
```

To tear down all resources:

```bash
cd smtj_batch_utils && python3 create_resources.py --clean
```

Refer to [smtj_batch_utils](./smtj_batch_utils/README.md) for the full configuration reference.

### Run the examples

The [examples](./examples) directory contains Jupyter notebooks demonstrating how to submit training jobs to the queues:

- [`notebook-fair-share.ipynb`](./examples/notebook-fair-share.ipynb) — Fair-share queue (share identifiers + weights)
- [`notebook-quota-management.ipynb`](./examples/notebook-quota-management.ipynb) — Quota management queue (quota shares + preemption)
