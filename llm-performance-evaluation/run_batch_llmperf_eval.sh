
#!/bin/bash

# List of model names in space-separated format
MODELS=('DeepSeek-R1-Distill-Qwen-1-5B-ml-g5-xlarge')

# Max input token test sizes
TOKEN_STEPS=(512 3072)

# Test endpoint for concurrent invocations
CONCURRENCY_STEPS=(1)

for CON in "${CONCURRENCY_STEPS[@]}"; do
    for TOKEN in "${TOKEN_STEPS[@]}"; do
        for MODEL_NAME in "${MODELS[@]}"; do
            RESULTS_DIR="output-results/results-concurrent-$CON-total-100/$MODEL_NAME--tok-$TOKEN--con-$CON"
        
            echo "Running ============> $MODEL_NAME || $TOKEN"
            # Create the results directory if it doesn't exist
            mkdir -p "$RESULTS_DIR"
            
            # Run the Python script with the provided model name
            python token_benchmark_ray.py \
                --model "$MODEL_NAME" \
                --mean-input-tokens $TOKEN \
                --stddev-input-tokens 32 \
                --mean-output-tokens 256 \
                --stddev-output-tokens 32 \
                --max-num-completed-requests 10 \
                --timeout 600 \
                --num-concurrent-requests $CON \
                --results-dir "$RESULTS_DIR" \
                --llm-api "sagemaker" \
                --additional-sampling-params '{}'
        done
    done
done
