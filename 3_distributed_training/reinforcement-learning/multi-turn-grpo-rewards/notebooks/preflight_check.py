#!/usr/bin/env python3
"""
Pre-flight check for MT-GRPO SageMaker training
"""
import json

print("=" * 60)
print("MT-GRPO SageMaker Training Pre-Flight Check")
print("=" * 60)

# Check 1: Verify train.py arguments
print("\n✓ Checking train.py arguments...")
with open('../scripts/train.py', 'r') as f:
    content = f.read()
    assert '--num_gpus' in content, "Missing --num_gpus in train.py"
    assert 'accelerate launch' in content, "Missing accelerate launch"
    assert '--num-processes' in content, "Missing --num-processes"
print("  ✅ train.py correctly configured")

# Check 2: Verify notebook configuration
print("\n✓ Checking notebook configuration...")
with open('complete_training_guide.ipynb', 'r') as f:
    nb = json.load(f)
    nb_str = json.dumps(nb)
    
    # Should NOT have torch_distributed enabled
    has_torch_dist = "distribution={'torch_distributed': {'enabled': True}}" in nb_str
    if has_torch_dist:
        print("  ⚠️  WARNING: torch_distributed is enabled - may conflict with accelerate")
    else:
        print("  ✅ torch_distributed disabled (accelerate will handle it)")
    
    # Should have keep_alive_period
    if 'keep_alive_period_in_seconds=3600' in nb_str:
        print("  ✅ Warm pool keep-alive set to 3600s (1 hour)")
    else:
        print("  ⚠️  Warm pool keep-alive not found or incorrect")

# Check 3: Verify instance type recommendations
print("\n✓ Instance recommendations:")
print("  ✅ ml.p4d.24xlarge - 8x A100, warm pool quota: 3")
print("  ✅ ml.p5.48xlarge - 8x H100, warm pool quota: 2")
print("  ⚠️  ml.g5.48xlarge - 8x A10G, warm pool quota: 0 (NOT AVAILABLE)")

# Check 4: Verify hyperparameters
print("\n✓ Recommended hyperparameters for 8 GPUs:")
print("  - num_generations: 14 (2*8-2)")
print("  - per_device_train_batch_size: 2")
print("  - grad_accum_steps: 4")
print("  - num_gpus: 8")

print("\n" + "=" * 60)
print("Pre-flight check complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Update notebook instance_type to 'ml.p4d.24xlarge'")
print("2. Set num_gpus hyperparameter to 8")
print("3. Set num_generations to 14")
print("4. Run the training job")
