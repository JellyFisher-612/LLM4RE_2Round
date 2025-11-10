#!/bin/bash
cd /root/autodl-tmp/LLaMA-Factory
# 统一基础路径
BASE_DIR="/home/users/lhy"

# 模型与 LoRA 路径
MODEL_PATH="/root/autodl-tmp/Llama-3.1-8B-Instruct"
LORA_PATH="/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/train_2025-11-9/checkpoint-4358"
EVAL_OUTPUT_DIR="/root/autodl-tmp/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/lora/dev_2025-11-9"

# # LLM4RE 路径
# LLM4RE_DIR="/home/users/lhy/LLM4RE_2Round"
# TEST_DATA_PATH="$LLM4RE_DIR/data/raw_data/test2.json"
# EXTRACTED_PATH="$LLM4RE_DIR/prediction/1030_3.json"
# FINAL_OUTPUT_PATH="$BASE_DIR/LLM4RE_v3/prediction/posted.json"
# LOG_PATH="$BASE_DIR/LLM4RE_v3/logs/output_process.log"

# 推理
llamafactory-cli train \
    --stage sft \
    --model_name_or_path "$MODEL_PATH" \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset step1_dev2 \
    --cutoff_len 2048 \
    --max_samples 100000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate True \
    --report_to none \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.7 \
    --output_dir "$EVAL_OUTPUT_DIR" \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --do_predict True \
    # --adapter_name_or_path "$LORA_PATH"

# # 提取预测
# python "$LLM4RE_DIR/extract_prediction.py" \
#     --predictions_path "$EVAL_OUTPUT_DIR/generated_predictions.jsonl" \
#     --test_data_path "$TEST_DATA_PATH" \
#     --output_path "$EXTRACTED_PATH"

# # LLM 后处理推理校验
# python "$LLM4RE_DIR/data_after_process.py" \
#     --test_data_path "$TEST_DATA_PATH" \
#     --prediction_path "$EXTRACTED_PATH" \
#     --model_path "$MODEL_PATH" \
#     --lora_path "$LORA_PATH" \
#     --output_path "$FINAL_OUTPUT_PATH" \
#     --log_path "$LOG_PATH" \