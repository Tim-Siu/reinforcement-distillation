#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
DATATYPES=("aime") # Default dataset
OUTPUT_DIR="$HOME"
N_PASSES=1
MAX_LENGTH=32768
TP_SIZE=1
N_GPUS_PER_NODE=8 # Default GPUs per node
DATA_SPLIT="data2" # Default data split directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n)
            N_PASSES="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --gpus-per-node)
            N_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --data-split)
            DATA_SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 \\"
            echo "    --model <model_path> \\"
            echo "    --datasets dataset1 dataset2 ... \\"
            echo "    --output-dir <output_directory> \\"
            echo "    --n <number_of_passes> \\"
            echo "    --max-length <max_response_length> \\"
            echo "    --tp <tensor_parallel_size> \\"
            echo "    --gpus-per-node <gpus_per_node> \\"
            echo "    --data-split <data_split_name>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of Passes: ${N_PASSES}"
echo "Max Response Length: ${MAX_LENGTH}"
echo "Tensor Parallel Size: ${TP_SIZE}"
echo "GPUs per Node: ${N_GPUS_PER_NODE}"
echo "Data Split: ${DATA_SPLIT}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        data.path=rllm/${DATA_SPLIT}/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=${N_PASSES} \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=${MAX_LENGTH} \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.8 \
        rollout.tensor_model_parallel_size=${TP_SIZE}
done