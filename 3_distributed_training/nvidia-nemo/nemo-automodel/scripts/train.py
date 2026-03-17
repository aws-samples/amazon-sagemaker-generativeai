# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# SageMaker entry point for NeMo Automodel fine-tuning.
# Runs the NeMo Automodel training recipe. All configuration (paths,
# hyperparameters, distributed strategy) is driven by the YAML config.

import logging

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    config = parse_args_and_load_config()

    logger.info("Starting NeMo Automodel fine-tuning recipe")
    recipe = TrainFinetuneRecipeForNextTokenPrediction(config)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
