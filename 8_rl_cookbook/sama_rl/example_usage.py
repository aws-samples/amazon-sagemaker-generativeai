#!/usr/bin/env python3
"""
Example: GRPO-RLVR Training with sama_rl

This example shows how to use GRPO with Reinforcement Learning from Verifiable Rewards (RLVR)
for mathematical reasoning tasks like GSM8K.
"""

from sama_rl import GRPO_RLVR, create_inference_model

def main():
    print("=== GRPO-RLVR Training Example with sama_rl ===")
    
    # Create GRPO-RLVR trainer
    rlvr_trainer = GRPO_RLVR(
        yaml_file="recipes/GRPO_RLVR/qwen2-0.5b-rlvr-config.yaml",
        instance_type="ml.g5.2xlarge",  # 24GB GPU for RLVR
        hf_token="your-hf-token-here",  # Required for model access
        max_steps=50  # Quick test
    )
    
    print("GRPO-RLVR trainer configured:")
    print(f"  Model: {rlvr_trainer.config.model['name']}")
    print(f"  Dataset: {rlvr_trainer.config.data['dataset_name']}")
    print(f"  Num shots: {rlvr_trainer.config.data['num_shots']}")
    print(f"  Instance: {rlvr_trainer.config.sagemaker['instance_type']}")
    print(f"  Max steps: {rlvr_trainer.config.training['max_steps']}")
    
    # Prepare GSM8K dataset
    print("\nPreparing GSM8K dataset...")
    # dataset = rlvr_trainer.prepare_dataset(
    #     dataset_name="gsm8k",
    #     num_shots=8,
    #     test_size=0.1
    # )
    
    # Upload dataset to S3
    # train_s3_path, val_s3_path = rlvr_trainer.upload_dataset_to_s3()
    # print(f"Dataset uploaded to S3: {train_s3_path}")
    
    # Start training (uncomment to actually run)
    # print("\nStarting GRPO-RLVR training...")
    # rlvr_trainer.train()
    # print(f"Training job: {rlvr_trainer.training_job_name}")
    
    # Get model artifacts (after training completes)
    # model_uri = rlvr_trainer.get_model_artifacts()
    # print(f"Model artifacts: {model_uri}")
    
    # Deploy for inference
    # inference_model = create_inference_model(
    #     model_uri=model_uri,
    #     instance_type="ml.g4dn.xlarge"
    # )
    
    # Test mathematical reasoning
    # test_problems = [
    #     "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    #     "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    #     "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
    # ]
    
    # for i, problem in enumerate(test_problems, 1):
    #     response = inference_model.predict(
    #         f"Question: {problem}\nLet me think step by step.\n",
    #         max_new_tokens=200
    #     )
    #     print(f"\nProblem {i}: {problem}")
    #     print(f"Solution: {response}")
    
    print("\n✅ GRPO-RLVR example configured successfully!")
    print("Uncomment the training lines to run actual RLVR training.")
    print("\nKey features:")
    print("- Verifiable rewards for mathematical reasoning")
    print("- GSM8K dataset with few-shot CoT prompting")
    print("- Step-by-step verification of mathematical solutions")
    print("- Integrated with sama_rl for easy deployment")

if __name__ == "__main__":
    main()
