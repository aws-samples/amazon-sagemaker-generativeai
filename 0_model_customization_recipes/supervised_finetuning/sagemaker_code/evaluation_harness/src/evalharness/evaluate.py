import mlflow
import pandas as pd
from bert_score import score as bert_score
from mlflow.metrics import (
    rouge1,
    rouge2,
    rougeL,
    rougeLsum,
    bleu,
    toxicity
)
from mlflow.metrics.genai import (
    answer_similarity,
    answer_correctness,
    answer_relevance,
    relevance,
    faithfulness
)
import logging

logging.basicConfig(level=logging.WARNING)


def get_run_id_from_name(experiment_name: str, run_name: str) -> str:
    # Look up the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search runs in that experiment by run_name (stored as tag)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if runs.empty:
        raise ValueError(f"No run found with name '{run_name}' in experiment '{experiment_name}'")

    # Take the first match (assuming run_name is unique per experiment)
    return runs.iloc[0].run_id


class Evaluate:
    def __init__(self, config, all_test_data):
        self.config = config
        self.all_test_data = all_test_data

        mlflow.set_tracking_uri(self.config.mlflow_tracking_server_arn)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        self.available_metrics = {
            'rouge1': rouge1(),
            'rouge2': rouge2(),
            'rougeL': rougeL(),
            'rougeLsum': rougeLsum(),
            'bleu': bleu(),
            'bert': bert_score,
            'toxicity': toxicity(),
            'answer_similarity': answer_similarity(
                model=self.config.model_judge,
                parameters=self.config.model_judge_parameters
            ),
            'answer_correctness': answer_correctness(
                model=self.config.model_judge,
                parameters=self.config.model_judge_parameters
            ),
            'answer_relevance': answer_relevance(
                model=self.config.model_judge,
                parameters=self.config.model_judge_parameters
            ),
            'relevance': relevance(
                model=self.config.model_judge,
                parameters=self.config.model_judge_parameters
            ),
            'faithfulness': faithfulness(
                model=self.config.model_judge,
                parameters=self.config.model_judge_parameters
            )
        }

        self.bert = False
        for metric in self.config.metrics:
            if metric not in self.available_metrics:
                logging.warning(f'{metric} is not a supported metric')
                self.config.metrics.remove(metric)
            if metric == 'bert':
                self.bert = True
                self.config.metrics.remove(metric)

    def run(self):

        # Lookup the existing parent run by experiment + run name
        parent_run_id = get_run_id_from_name(
            experiment_name=self.config.mlflow_experiment_name,
            run_name=self.config.mlflow_run_name
        )
        
        # Attach to the existing parent run
        with mlflow.start_run(run_id=parent_run_id):
            for key, test_data in self.all_test_data.items():
                if key == 'source_model_test_data':
                    run_name = self.config.source_model_predictions_path.split('.jsonl')[0]
                elif key == 'target_model_test_data':
                    run_name = self.config.target_model_predictions_path.split('.jsonl')[0]
                elif key == 'reference_model_test_data':
                    run_name = self.config.reference_model_predictions_path.split('.jsonl')[0]
                else:
                    run_name = 'unknown_name'
    
                # Start nested run under the parent
                with mlflow.start_run(run_name=run_name, nested=True):
                    test_data_df = pd.DataFrame(test_data)
    
                    mlflow.evaluate(
                        data=test_data_df,
                        predictions='model_predictions',
                        targets='ground_truth',
                        evaluators='default',
                        extra_metrics=[
                            self.available_metrics[metric]
                            for metric in self.config.metrics
                        ]
                    )
    
                    if self.bert:
                        precision, recall, f1 = bert_score(
                            test_data_df.model_predictions.tolist(),
                            test_data_df.ground_truth.tolist(),
                            lang='en',
                            verbose=False
                        )
                        mlflow.log_metrics({
                            'bert_precision': precision.mean(),
                            'bert_recall': recall.mean(),
                            'bert_f1': f1.mean()
                        })
