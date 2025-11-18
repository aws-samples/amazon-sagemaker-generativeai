import click

from .config import Config
from .evaluate import Evaluate
from .test_data import TestData


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default="evalharness.yaml",
    help="Path to eval configuration file. Default is evalharness.yaml",
    show_default=True,
)
def cli(config):
    try:
        # Load config
        config = Config.load(config)

        # Load test data if available
        test_data = TestData()
        all_test_data = {}
        if config.source_model_predictions_path:
            source_model_test_data = test_data.load(config.source_model_predictions_path)
            all_test_data['source_model_test_data'] = source_model_test_data
        if config.target_model_predictions_path:
            target_model_test_data = test_data.load(config.target_model_predictions_path)
            all_test_data['target_model_test_data'] = target_model_test_data
        if config.reference_model_predictions_path:
            reference_model_test_data = test_data.load(config.reference_model_predictions_path)
            all_test_data['reference_model_test_data'] = reference_model_test_data

        # Evaluate all datasets
        evaluate = Evaluate(config, all_test_data)
        evaluate.run()
    except Exception as e:
        raise e

