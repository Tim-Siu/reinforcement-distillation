NUM_GPUS=8
MODEL_PATH=data/Qwen-1.5B-Math-REDI-53k
MODEL_ARGS="pretrained=$MODEL_PATH,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL_PATH

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../openr1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details

# AIME 2024
TASK=aime25
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../openr1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../openr1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks ../openr1/src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
