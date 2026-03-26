import argparse

from config import batch_client, load_config
from queue_manager import TrainingQueueManager, create_resources, delete_resources


def main():
    parser = argparse.ArgumentParser(
        description="Create or delete AWS Batch resources for SageMaker Training job queuing."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all resources defined in config.yaml",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: config.yaml in this directory)",
    )
    args = parser.parse_args()

    queue_configs = load_config(args.config)
    manager = TrainingQueueManager(batch_client, verbose=not args.clean)

    resources = create_resources(queue_configs, manager)

    if args.clean:
        manager.verbose = True
        delete_resources(resources, manager)


if __name__ == "__main__":
    main()
