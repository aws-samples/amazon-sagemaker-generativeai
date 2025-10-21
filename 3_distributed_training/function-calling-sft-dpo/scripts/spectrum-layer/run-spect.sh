git clone https://github.com/cognitivecomputations/spectrum.git
cd spectrum
# generate yaml configuration
#python3 spectrum.py --model-name meta-llama/Meta-Llama-3.1-8B --top-percent 30
python3 spectrum.py --model-name Qwen/Qwen3-1.7B --top-percent 10
# Top 30% SNR layers saved to snr_results_meta-llama-Meta-Llama-3.1-8B_unfrozenparameters_30percent.yaml
cd ..