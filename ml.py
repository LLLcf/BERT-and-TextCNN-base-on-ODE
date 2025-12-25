import os
import re
import json
import logging
import random
import warnings
import numpy as np
import pandas as pd
import jieba
import torch
from tqdm import tqdm

# Sklearn 核心组件
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Transformers
from transformers import BertTokenizer, BertModel

# --- 1. 全局配置 ---
warnings.filterwarnings("ignore")

# 路径配置
DATA_PATH = "../data/paper_full_data.jsonl"
BERT_PATH = "/root/lanyun-fs/models/Bert"
LOG_DIR = "../log"

# 确保日志目录存在
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, "sklearn_experiment_results.log")

# 配置日志
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='w'), # mode='w' 每次覆盖，避免重复追加
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 随机种子
SEED = 2025

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

setup_seed(SEED)

# 停用词 (示例)
STOPWORDS = set([
    '\n', '\t', ' ', '　', ',', '.', '!', '?', ';', ':', '、', '，', '。', '！', '？',
    '的', '了', '是', '在', '和', '有', '就', '都', '而', '及', '与', '这', '那'
])

# --- 2. 数据处理工具 ---

def clean_text(text):
    text = str(text)
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text.strip()

def chinese_tokenizer(text):
    """TF-IDF 专用分词器"""
    text = clean_text(text)
    words = jieba.lcut(text)
    return [word for word in words if word not in STOPWORDS and len(word) > 1]

def load_and_process_dataframe(file_path):
    """加载数据 -> 分层下采样 -> 标签编码"""
    logger.info(f"正在加载数据: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
        
    # 读取 JSONL
    df = pd.read_json(file_path, lines=True)
    logger.info(f"原始数据量: {len(df)}")
    
    # 确保有需要的列
    if 'content' not in df.columns or '类别' not in df.columns:
        raise ValueError("数据集缺少 'content' 或 '类别' 列")

    # === 1. 分层下采样 (1/5) ===
    logger.info("执行分层下采样 (保留 1/5)...")
    # 使用 groupby + sample 保证各类别比例不变
    df = df.groupby('类别', group_keys=False).apply(
        lambda x: x.sample(frac=1/5, random_state=SEED)
    ).reset_index(drop=True)
    logger.info(f"采样后数据量: {len(df)}")
    # ==========================

    # === 2. 标签编码 ===
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['类别'])
    
    logger.info(f"类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return df, le.classes_

# --- 3. 特征工程: BERT Embedding ---

class BertVectorizor:
    """从本地路径加载 BERT 并提取特征"""
    def __init__(self, model_path, batch_size=32, device=None):
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"正在从本地加载 BERT: {model_path} (Device: {self.device})")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"BERT 加载失败: {e}")
            raise e

    def transform(self, texts):
        logger.info(f"正在提取 BERT 特征 (共 {len(texts)} 条)...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT Embedding"):
            batch_texts = texts[i : i + self.batch_size]
            batch_texts = [clean_text(t) for t in batch_texts] # 简单清洗
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                # 取 [CLS] token (batch, hidden_size)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings)

# --- 4. 模型定义与实验流程 ---

def get_models():
    """定义传统机器学习模型"""
    return {
        'LogisticRegression': LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=SEED),
        'SVM': SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=SEED), 
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=SEED, n_jobs=-1)
    }

def run_experiment(exp_name, X_train, X_test, y_train, y_test, models, target_names):
    """运行一组模型实验"""
    logger.info(f"\n{'#'*40}\n开始实验: {exp_name}\n{'#'*40}")
    results = {} # 存储 F1 分数
    
    for name, model in models.items():
        logger.info(f"[{exp_name}] 正在训练模型: {name} ...")
        start_time = pd.Timestamp.now()
        try:
            # 训练
            model.fit(X_train, y_train)
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro') # 使用 Macro F1
            report = classification_report(y_test, y_pred, target_names=target_names)
            
            results[name] = f1
            
            # 记录详细日志
            log_msg = (
                f"\n--- 实验: {exp_name} | 模型: {name} ---\n"
                f"耗时: {pd.Timestamp.now() - start_time}\n"
                f"Macro F1: {f1:.4f} (Acc: {acc:.4f})\n"
                f"分类报告:\n{report}\n"
            )
            logger.info(log_msg)
            
        except Exception as e:
            logger.error(f"[{exp_name}] 模型 {name} 失败: {e}")
            
    return results

# --- 5. 主函数 ---

def main():
    try:
        # 1. 加载并处理 DataFrame
        df, label_names = load_and_process_dataframe(DATA_PATH)
        
        # 2. 数据划分 (Train/Val/Test = 8:1:1) - 分层划分
        logger.info("正在进行数据集划分 (Train:Val:Test = 8:1:1)...")
        
        # 第一步：划分出 训练集(80%) 和 剩余集(20%)
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['label_id'], 
            random_state=SEED
        )
        
        # 第二步：将 剩余集 对半划分为 验证集(10%) 和 测试集(10%)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df['label_id'],
            random_state=SEED
        )
        
        logger.info(f"数据集规模 -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # 提取数据列表（方便后续特征提取）
        text_train = train_df['content'].tolist()
        y_train = train_df['label_id'].values
        
        text_test = test_df['content'].tolist()
        y_test = test_df['label_id'].values

        # ==========================================
        # 实验一: TF-IDF + 多模型
        # ==========================================
        logger.info("构建 TF-IDF 特征...")
        tfidf_vec = TfidfVectorizer(
            tokenizer=chinese_tokenizer,
            max_features=5000,
            ngram_range=(1, 2)
        )
        # fit 只能在训练集上进行
        X_train_tfidf = tfidf_vec.fit_transform(text_train)
        X_test_tfidf = tfidf_vec.transform(text_test)
        
        res_tfidf = run_experiment(
            "TF-IDF特征", 
            X_train_tfidf, X_test_tfidf, y_train, y_test, 
            get_models(), label_names
        )
        
        # ==========================================
        # 实验二: BERT Embedding + 多模型
        # ==========================================
        logger.info("构建 BERT Embedding 特征...")
        if os.path.exists(BERT_PATH):
            bert_vec = BertVectorizor(model_path=BERT_PATH)
            
            # 提取向量 (Dense Features)
            X_train_bert = bert_vec.transform(text_train)
            X_test_bert = bert_vec.transform(text_test)
            
            logger.info(f"BERT 特征维度: {X_train_bert.shape}")
            
            res_bert = run_experiment(
                "BERT嵌入特征", 
                X_train_bert, X_test_bert, y_train, y_test, 
                get_models(), label_names
            )
        else:
            logger.error(f"BERT路径不存在: {BERT_PATH}，跳过实验二")
            res_bert = {}
            
        # ==========================================
        # 总结
        # ==========================================
        logger.info("\n" + "="*50)
        logger.info("所有实验结束，结果汇总 (Macro F1):")
        logger.info("="*50)

        logger.info("--- 实验一: TF-IDF ---")
        if res_tfidf:
            best_model = max(res_tfidf, key=res_tfidf.get)
            for k, v in res_tfidf.items():
                logger.info(f"{k}: {v:.4f}")
            logger.info(f"Best: {best_model} ({res_tfidf[best_model]:.4f})")

        logger.info("\n--- 实验二: BERT Embedding ---")
        if res_bert:
            best_model = max(res_bert, key=res_bert.get)
            for k, v in res_bert.items():
                logger.info(f"{k}: {v:.4f}")
            logger.info(f"Best: {best_model} ({res_bert[best_model]:.4f})")
            
        logger.info(f"\n日志文件路径: {LOG_FILE}")

    except Exception as e:
        logger.error(f"程序运行发生致命错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()