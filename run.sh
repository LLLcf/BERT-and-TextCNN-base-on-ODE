#!/bin/bash

DATA_PATH="../data/paper_full_data.jsonl"
BERT_PATH="/root/lanyun-fs/models/Bert"
# 确保目录存在
mkdir -p "../log"
RESULT_LOG="../log/experiment_results.log"

# 初始化主日志
echo "=============================================" > $RESULT_LOG
echo "MASTER EXPERIMENT REPORT (Unified Train Script)" >> $RESULT_LOG
echo "Date: $(date)" >> $RESULT_LOG
echo "---------------------------------------------" >> $RESULT_LOG

# ==========================================================
# PART 0: 运行 Sklearn 机器学习基准 (Integrated in train.py)
# ==========================================================
echo "=================================================="
echo "Running Part 0: Scikit-Learn Baselines"
echo "=================================================="

# 调用 train.py 的 sklearn 模式
python -u train.py --model_type sklearn --data_path ${DATA_PATH} --bert_path ${BERT_PATH} >> $RESULT_LOG

echo "Sklearn experiments completed. Results appended to master log."

# ==========================================================
# PART 1: 运行传统深度学习模型 (Traditional TextCNN)
# ==========================================================
echo "=================================================="
echo "Running Part 1: Deep Learning - Traditional (Scratch)"
echo "=================================================="

# 传统模型通常需要更多 epoch，学习率较大
CMD="python -u train.py --model_type traditional --data_path ${DATA_PATH} --bert_path ${BERT_PATH} --lr 1e-4 --epochs 20"

# 执行训练并打印输出
OUTPUT=$($CMD)
echo "$OUTPUT"

# 提取保存的模型路径
SAVED_MODEL=$(echo "$OUTPUT" | grep "MODEL_SAVED_AT:" | cut -d':' -f2)

if [ ! -z "$SAVED_MODEL" ]; then
    echo "Prediction for traditional..."
    PRED_CMD="python -u predict.py --model_type traditional --model_path ${SAVED_MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH}"
    
    echo "---------------------------------------------" >> $RESULT_LOG
    echo "Experiment: traditional [scratch]" >> $RESULT_LOG
    echo "Model Path: ${SAVED_MODEL}" >> $RESULT_LOG
    $PRED_CMD >> $RESULT_LOG
    echo "" >> $RESULT_LOG
else
    echo "Error: Traditional model training failed."
fi

# ==========================================================
# PART 2: 运行 BERT 系列模型 (Frozen & Finetune)
# ==========================================================
for MODEL in bert textcnn node; do
    for MODE in frozen finetune; do
        
        echo "=================================================="
        echo "Running Part 2: Model=${MODEL}, Mode=${MODE}"
        echo "=================================================="
        
        CMD="python -u train.py --model_type ${MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH} --epochs 10"
        
        if [ "$MODE" == "finetune" ]; then
            CMD="$CMD --fine_tune --lr 2e-5"
        else
            CMD="$CMD --lr 2e-4"
        fi
        
        OUTPUT=$($CMD)
        echo "$OUTPUT"
        
        SAVED_MODEL=$(echo "$OUTPUT" | grep "MODEL_SAVED_AT:" | cut -d':' -f2)
        
        if [ -z "$SAVED_MODEL" ]; then
            echo "Error: Training failed for ${MODEL} ${MODE}"
            continue
        fi
        
        echo "Prediction for ${MODEL} ${MODE}..."
        PRED_CMD="python -u predict.py --model_type ${MODEL} --model_path ${SAVED_MODEL} --data_path ${DATA_PATH} --bert_path ${BERT_PATH}"
        
        if [ "$MODE" == "finetune" ]; then
            PRED_CMD="$PRED_CMD --fine_tune"
        fi
        
        echo "---------------------------------------------" >> $RESULT_LOG
        echo "Experiment: ${MODEL} [${MODE}]" >> $RESULT_LOG
        echo "Model Path: ${SAVED_MODEL}" >> $RESULT_LOG
        $PRED_CMD >> $RESULT_LOG
        echo "" >> $RESULT_LOG
        
    done
done

echo "All experiments finished! Full report saved to $RESULT_LOG"