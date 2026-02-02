import verifiers as vf

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.MathEnv(dataset="math")
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

run_name = "math_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()