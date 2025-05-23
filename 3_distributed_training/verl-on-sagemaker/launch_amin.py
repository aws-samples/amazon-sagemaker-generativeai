from sagemaker.modules.train import ModelTrainer
from sagemaker.modules.configs import SourceCode, Compute, InputData
from sagemaker.modules.train.model_trainer import Mode
from huggingface_hub import HfFolder
from dotenv import load_dotenv
import os
import boto3
from sagemaker.modules import Session

sess = Session(boto3.session.Session(region_name='us-east-1'))
iam = boto3.client('iam')
role = iam.get_role(RoleName='sagemaker')['Role']['Arn']

load_dotenv()

# image URI for the training job
# pytorch_image = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker"
verl_image = '783764584149.dkr.ecr.us-east-1.amazonaws.com/sagemaker-verl-training:verl-py310-torch26-vllm083'
# you can find all available images here
# https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths.html

env = {
    'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
    'HF_TOKEN': HfFolder.get_token(),
}

# define the script to be run
source_code = SourceCode(
    source_dir="scripts",
    # requirements="requirements.txt",
    # entry_script="unsloth_train.py",
    # command='chmod +x examples/grpo/run_qwen2-7b.sh && ./examples/grpo/run_qwen2-7b.sh'
    command='bash ./examples/grpo_trainer/run_qwen2-7b.sh'
    # command='python3 /opt/ml/input/data/code/verl/examples/data_preprocess/gsm8k.py --local_dir /temp/data/gsm8k',
    # command='''python3 -m verl.trainer.main_ppo \
    # algorithm.adv_estimator=grpo \
    # data.train_files=gsm8k/train.parquet \
    # data.val_files=gsm8k/test.parquet \
    # data.train_batch_size=1024 \
    # data.max_prompt_length=512 \
    # data.max_response_length=1024 \
    # data.filter_overlong_prompts=True \
    # data.truncation='error' \
    # actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    # actor_rollout_ref.actor.optim.lr=1e-6 \
    # actor_rollout_ref.model.use_remove_padding=True \
    # actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    # actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40 \
    # actor_rollout_ref.actor.use_kl_loss=True \
    # actor_rollout_ref.actor.kl_loss_coef=0.001 \
    # actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    # actor_rollout_ref.actor.entropy_coeff=0 \
    # actor_rollout_ref.model.enable_gradient_checkpointing=True \
    # actor_rollout_ref.actor.fsdp_config.param_offload=False \
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    # actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    # actor_rollout_ref.rollout.name=vllm \
    # actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    # actor_rollout_ref.rollout.n=5 \
    # actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    # actor_rollout_ref.ref.fsdp_config.param_offload=True \
    # algorithm.use_kl_in_reward=False \
    # trainer.critic_warmup=0 \
    # trainer.logger=['console','wandb'] \
    # trainer.project_name='verl' \
    # trainer.experiment_name='qwen2_7b_function_rm' \
    # trainer.n_gpus_per_node=8 \
    # trainer.nnodes=1 \
    # trainer.save_freq=20 \
    # trainer.test_freq=5 \
    # trainer.total_epochs=15 $@''',
    # command='python3 /opt/ml/input/data/code/list_dir.py'
)

# Compute configuration for the training job
compute = Compute(
    instance_count=1,
    # instance_type='local_gpu',
    instance_type="ml.p5.48xlarge",
    # instance_type="ml.p4d.24xlarge",
    volume_size_in_gb=96,
    keep_alive_period_in_seconds=3600,
)

# define the ModelTrainer
model_trainer = ModelTrainer(
    sagemaker_session=sess,
    training_image=verl_image,
    source_code=source_code,
    base_job_name="verl-grpo-example",
    compute=compute,
    environment=env,
    role=role,
    # training_mode=Mode.LOCAL_CONTAINER,
    # local_container_root="temp"
)

# pass the input data
# input_data = InputData(
#     channel_name="train",
#     data_source=training_input_path,  #s3 path where training data is stored
# )

# start the training job
model_trainer.train(wait=True)