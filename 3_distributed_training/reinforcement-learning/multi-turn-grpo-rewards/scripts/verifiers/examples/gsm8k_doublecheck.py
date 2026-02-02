import verifiers as vf

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.DoubleCheckEnv(dataset="gsm8k")
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

run_name = "gsm8k-dc_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8, reward_weights=vf_env.rubric.get_reward_weights())
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()