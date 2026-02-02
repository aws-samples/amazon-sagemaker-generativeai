import verifiers as vf
from verifiers.prompts import CODE_PROMPT, CODE_FEW_SHOT

#model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

vf_env = vf.CodeEnv(
    dataset="math",
    few_shot=CODE_FEW_SHOT[0],
    system_prompt=CODE_PROMPT
)
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "math-code_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8)
# # rollouts per prompt
# training_args.num_generations = 7
# # minibatch size per GPU ( bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
# training_args.per_device_train_batch_size = 6
# # batches to accumulate (6 prompts * 4 -> 24 prompts per global batch)
# training_args.gradient_accumulation_steps = 4
# # steps per global batch (1 on-policy, 1 off-policy)
# training_args.num_iterations = 2
# # ref model weight
# training_args.beta = 0.04
# training_args.max_grad_norm = 0.1

training_args.learning_rate = 2e-6
# rollouts per prompt
training_args.num_generations = 21
# minibatch size per GPU ( bs 12 * 7 gpus / 21 rollouts -> 4 prompts per batch)
training_args.per_device_train_batch_size = 12
# batches to accumulate (4 prompts * 2 -> 8 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2

training_args.max_steps = 200
training_args.beta = 0.02

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset
)
trainer.train()