#!/bin/bash

DATA_PATH="../data/paper_full_data.jsonl"
BERT_PATH="/root/lanyun-fs/models/Bert"
# 确保目录存在
mkdir -p "../log"
mkdir -p "../checkpoints"
RESULT_LOG="../log/experiment_results_new2.log"

# 初始化主日志
echo "=============================================" > $RESULT_LOG
echo "MASTER EXPERIMENT REPORT (Unified Train Script)" >> $RESULT_LOG
echo "Date: $(date)" >> $RESULT_LOG
echo "---------------------------------------------" >> $RESULT_LOG

# # ==========================================================
# # PART 0: 运行 Sklearn 机器学习基准
# # ==========================================================
# echo "=================================================="
# echo "Running Part 0: Scikit-Learn Baselines"
# echo "=================================================="

# # 调用 train.py 的 sklearn 模式
# python -u train.py --model_type sklearn --data_path ${DATA_PATH} --bert_path ${BERT_PATH} >> $RESULT_LOG

# echo "Sklearn experiments completed. Results appended to master log."


# # ==========================================================
# # PART 1: 运行基于词表的深度学习模型 (Vocab/Embedding Based)
# # 模型: traditional, node_textcnn, node_textcnn_paral
# # 特点: 从零训练 Embedding, 需要较大学习率和更多 Epoch
# # ==========================================================
# # VOCAB_MODELS="traditional node_textcnn node_textcnn_paral"
VOCAB_MODELS="node_textcnn node_textcnn_paral"


for MODEL in $VOCAB_MODELS; do
    echo "=================================================="
    echo "Running Part 1: Deep Learning - ${MODEL} (Scratch)"
    echo "=================================================="

    # 词向量模型通常需要更多 epoch (20)，学习率较大 (1e-3)
    CMD="python -u train.py --model_type ${MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH} --lr 1e-3 --epochs 20"

    # 执行训练并捕获输出
    OUTPUT=$($CMD)
    # 将训练过程打印到终端以便观察
    echo "$OUTPUT"

    # 提取保存的模型路径 (处理可能存在的空格)
    SAVED_MODEL=$(echo "$OUTPUT" | grep "MODEL_SAVED_AT:" | awk -F': ' '{print $2}' | tr -d '\r')

    if [ ! -z "$SAVED_MODEL" ] && [ -f "$SAVED_MODEL" ]; then
        echo "Prediction for ${MODEL}..."
        # 预测命令
        PRED_CMD="python -u predict.py --model_type ${MODEL} --model_path ${SAVED_MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH}"
        
        echo "---------------------------------------------" >> $RESULT_LOG
        echo "Experiment: ${MODEL} [scratch]" >> $RESULT_LOG
        echo "Model Path: ${SAVED_MODEL}" >> $RESULT_LOG
        $PRED_CMD >> $RESULT_LOG
        echo "" >> $RESULT_LOG
    else
        echo "Error: ${MODEL} training failed or model file not found."
    fi
done


# ==========================================================
# PART 2: 运行 BERT 系列模型 (Frozen & Finetune)
# 模型: bert, textcnn, bert_node_textcnn, bert_node_textcnn_paral
# ==========================================================
# BERT_MODELS="bert textcnn bert_node_textcnn bert_node_textcnn_paral"
# BERT_MODELS="bert_node_textcnn_paral"


# for MODEL in $BERT_MODELS; do
#     for MODE in frozen finetune; do
        
#         echo "=================================================="
#         echo "Running Part 2: Model=${MODEL}, Mode=${MODE}"
#         echo "=================================================="
        
#         # 基础命令
#         CMD="python -u train.py --model_type ${MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH} --epochs 10"
        
#         # 根据模式调整参数
#         if [ "$MODE" == "finetune" ]; then
#             # 微调: 学习率小 (2e-5)
#             CMD="$CMD --fine_tune --lr 2e-5"
#         else
#             # 冻结: 学习率稍大 (2e-4) 以训练下游分类层
#             CMD="$CMD --lr 2e-4"
#         fi
        
#         OUTPUT=$($CMD)
#         echo "$OUTPUT"
        
#         SAVED_MODEL=$(echo "$OUTPUT" | grep "MODEL_SAVED_AT:" | awk -F': ' '{print $2}' | tr -d '\r')

#         if [ ! -z "$SAVED_MODEL" ] && [ -f "$SAVED_MODEL" ]; then
#             echo "Prediction for ${MODEL} ${MODE}..."
            
#             PRED_CMD="python -u predict.py --model_type ${MODEL} --model_path ${SAVED_MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH}"
            
#             if [ "$MODE" == "finetune" ]; then
#                 PRED_CMD="$PRED_CMD --fine_tune"
#             fi
            
#             echo "---------------------------------------------" >> $RESULT_LOG
#             echo "Experiment: ${MODEL} [${MODE}]" >> $RESULT_LOG
#             echo "Model Path: ${SAVED_MODEL}" >> $RESULT_LOG
#             $PRED_CMD >> $RESULT_LOG
#             echo "" >> $RESULT_LOG
#         else
#             echo "Error: Training failed for ${MODEL} ${MODE}"
#             continue
#         fi
        
#     done
# done

echo "All experiments finished! Full report saved to $RESULT_LOG"