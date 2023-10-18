""" SageMaker Experiments callback implementation"""

import importlib
import logging
from transformers import TrainerCallback


# disable INFO and WARNING logging status to prevent flood of WARNs
logging.getLogger("sagemaker").setLevel(logging.CRITICAL)


def is_sagemaker_available():
    return importlib.util.find_spec("sagemaker") is not None


class SageMakerExperimentsCallback(TrainerCallback):
    """
    SageMaker Experiments Plus transformer callback. 
    Designed to allow auto logging from transformer API.
    """
    def __init__(
        self, 
        region,
        _has_sagemaker_experiments=is_sagemaker_available()
    ):
        
        assert (
            _has_sagemaker_experiments
        ), "SageMakerExperimentsCallback requires sagemaker to be install. Run 'pip install -U sagemaker'"
        
        import boto3
        import sagemaker
        from sagemaker.experiments.run import load_run      
        
        self.sagemaker_session = sagemaker.session.Session(
            boto3.session.Session(region_name=region)
        )
        self.local_load_run = load_run
        
        # epoch tracker
        self.last_epoch = None
        
        with load_run(sagemaker_session=self.sagemaker_session) as run: 
            self.sm_experiments_run = run
            self.ctx_exp_name = run.experiment_name
            self.ctx_run_name = run.run_name
            
            print(f"[sm-callback] loaded sagemaker Experiment (name: {self.ctx_exp_name}) with run: {self.ctx_run_name}!")
    
    def on_init_end(self, args, state, control, **kwargs):
        
        print(f"[sm-callback] adding parameters to {self.ctx_exp_name}: {self.ctx_run_name}")
        
        with self.local_load_run(
            experiment_name=self.ctx_exp_name, 
            run_name=self.ctx_run_name,
            sagemaker_session=self.sagemaker_session
        ) as ctx_run: 
            ctx_run.log_parameters(
                {
                    k: str(v) if str(v) else None 
                        for k, v in vars(args).items() 
                            if isinstance(v, (str, int, float, bool))
                }
            )
              
    def on_log(self, args, state, control, logs=None, **kwargs):
        
        with self.local_load_run(
            experiment_name=self.ctx_exp_name, 
            run_name=self.ctx_run_name,
            sagemaker_session=self.sagemaker_session
        ) as ctx_run: 
            
            for k, v in logs.items():
                if not k.startswith('eval'):
                    ctx_run.log_metric(
                        name=f"train/step:{k}", 
                        value=v, 
                        step=int(state.global_step)
                    )

    def on_epoch_end(self, args, state, control, logs=None, **kwargs): 
        """
        On epoch end we average results and log it into an epoch value as x 
        and average of metrics as y
        """
        with self.local_load_run(
            experiment_name=self.ctx_exp_name, 
            run_name=self.ctx_run_name,
            sagemaker_session=self.sagemaker_session
        ) as ctx_run:
            
            epoch_history = state.log_history
            
            if self.last_epoch is None:
                self.last_epoch = 0
            
            current_epoch = int(round(epoch_history[-1]['epoch']))
            
            print(f"[sm-callback] start: {self.last_epoch} ep to end: {current_epoch} ep!")
            
            epoch_loss_values = {
                row['epoch']: row['loss'] 
                for row in epoch_history 
                if self.last_epoch < row['epoch'] <= current_epoch
            }
            average_epoch_loss = sum(list(epoch_loss_values.values()))/len(epoch_loss_values)
            
            ctx_run.log_metric(
                name="train/epoch:loss",
                value=average_epoch_loss, 
                step=int(current_epoch)
            )
            
            self.last_epoch = current_epoch

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """
        On train end we average results and log it into an epoch value as x 
        and average of metrics as y
        """
        with self.local_load_run(
            experiment_name=self.ctx_exp_name, 
            run_name=self.ctx_run_name,
            sagemaker_session=self.sagemaker_session
        ) as ctx_run:
            
            epoch_history = state.log_history
            
            ctx_run.log_metric(
                name="final/eval:loss",
                value=epoch_history[-1]["eval_loss"] 
            )