import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from tqdm import tqdm
import jieba
import pandas as pd 
import warnings
import numpy as np
import random

# --- 引入自定义模块 ---
# 请确保 model_utils.py 中包含了以下所有类
from model_utils import (
    setup_seed, jsonl_to_dataframe, 
    THUNewsDataset, TraditionalDataset, Vocab,
    BertClassifier, BERT_TextCNN, TraditionalTextCNN,
    # BERT 版 NODE 模型
    BERT_NODE_TEXTCNN, BERT_NODE_TEXTCNN_PARALLEL, 
    # 纯 Embedding 版 NODE 模型 (本次新增)
    NODE_TEXTCNN, NODE_TEXTCNN_PARALLEL,
    BertVectorizor, chinese_tokenizer, clean_text
)

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser(description="Training Script")
    # 路径
    parser.add_argument("--data_path", type=str, default="../data/paper_full_data.jsonl")
    parser.add_argument("--bert_path", type=str, default="/root/lanyun-fs/models/Bert")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/")
    
    # 核心实验参数
    # 更新了 model_type 的选择范围
    parser.add_argument("--model_type", type=str, required=True,
                        choices=[
                            'sklearn', 
                            # 传统/词向量模型
                            'traditional', 'node_textcnn', 'node_textcnn_paral',
                            # BERT 基座模型
                            'bert', 'textcnn', 'bert_node_textcnn', 'bert_node_textcnn_paral'
                        ])
    parser.add_argument("--fine_tune", action="store_true", help="是否微调BERT")
    
    # 超参
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32, help="BERT系列模型的BatchSize")
    parser.add_argument("--batch_size_traditional", type=int, default=64, help="非BERT模型的BatchSize")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--embed_dim", type=int, default=512, help="非BERT模型的词向量维度")
    parser.add_argument("--ode_step", type=int, default=11, help="Neural ODE 的积分步数")
    
    return parser.parse_args()

def run_sklearn_experiments(train_df, test_df, label_names, bert_path, seed):
    """ Sklearn 机器学习基准实验 (TF-IDF & BERT Embeddings) """
    print(f"\n{'='*20} Running Sklearn Baselines {'='*20}")
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    text_train = train_df['content'].tolist()
    text_test = test_df['content'].tolist()
    
    def get_sklearn_models():
        return {
            'LogisticRegression': LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=seed),
            'SVM': SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=seed), 
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed, n_jobs=-1),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=seed, n_jobs=-1)
        }

    # 1. TF-IDF
    print("\n>>> Feature: TF-IDF")
    tfidf_vec = TfidfVectorizer(tokenizer=chinese_tokenizer, max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vec.fit_transform(text_train)
    X_test_tfidf = tfidf_vec.transform(text_test)
    
    for name, model in get_sklearn_models().items():
        print(f"Training {name} (TF-IDF)...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        print(f"[{name}] Macro F1: {f1_score(y_test, y_pred, average='macro'):.6f}")

    # 2. BERT Embedding
    print("\n>>> Feature: BERT Embedding (Frozen)")
    if os.path.exists(bert_path):
        bert_vec = BertVectorizor(model_path=bert_path)
        X_train_bert = bert_vec.transform(text_train)
        X_test_bert = bert_vec.transform(text_test)
        
        for name, model in get_sklearn_models().items():
            print(f"Training {name} (BERT-Embed)...")
            model.fit(X_train_bert, y_train)
            y_pred = model.predict(X_test_bert)
            print(f"[{name}] Macro F1: {f1_score(y_test, y_pred, average='macro'):.6f}")
    else:
        print(f"Error: BERT Path not found {bert_path}, skipping.")

def train(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 定义模型分组 ---
    # 词表类模型：使用 Jieba 分词 + 随机初始化 Embedding
    VOCAB_MODELS = ['traditional', 'node_textcnn', 'node_textcnn_paral']
    # BERT类模型：使用 BertTokenizer + Pretrained BERT
    BERT_MODELS = ['bert', 'textcnn', 'bert_node_textcnn', 'bert_node_textcnn_paral']

    # --- 1. 数据加载与划分 ---
    df = jsonl_to_dataframe(args.data_path)
    print(f"Original Dataset Size: {len(df)}")

    print("Performing Stratified Downsampling (keeping 1/5)...")
    df = df.groupby('类别', group_keys=False).apply(
        lambda x: x.sample(frac=1/5, random_state=args.seed)
    ).reset_index(drop=True)
    
    labels = sorted(list(set(df['类别'])))
    label2id = {cat: ids for ids, cat in enumerate(labels)}
    df['label'] = df['类别'].apply(lambda x: label2id[str(x)])
    num_classes = len(labels)
    
    # 划分 (Train 80% / Val 10% / Test 10%)
    train_data = df.sample(frac=0.8, random_state=args.seed)
    remaining_data = df.drop(train_data.index)
    val_data = remaining_data.sample(frac=0.5, random_state=args.seed)
    test_data = remaining_data.drop(val_data.index).reset_index(drop=True)
    
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    if args.model_type == 'sklearn':
        run_sklearn_experiments(train_data, test_data, labels, args.bert_path, args.seed)
        return 

    # --- 2. 深度学习模型初始化 ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # A. 词表类模型构建流程 (Traditional / NODE_TEXTCNN / NODE_TEXTCNN_PARALLEL)
    if args.model_type in VOCAB_MODELS:
        mode_name = "scratch"
        current_batch_size = args.batch_size_traditional
        
        print(f"Building Vocabulary for {args.model_type}...")
        all_text = train_data['content'].tolist()
        all_tokens = []
        for text in tqdm(all_text, desc="Tokenizing"):
            all_tokens.extend(jieba.lcut(text))
        
        vocab = Vocab(all_tokens, min_freq=2)
        vocab.save(os.path.join(args.output_dir, f"{args.model_type}_vocab.json"))
        print(f"Vocab Size: {len(vocab)}")
        
        train_dataset = TraditionalDataset(train_data, vocab, 512)
        val_dataset = TraditionalDataset(val_data, vocab, 512)
        
        # 模型选择
        if args.model_type == 'traditional':
            model = TraditionalTextCNN(len(vocab), args.embed_dim, num_classes).to(device)
        elif args.model_type == 'node_textcnn':
            model = NODE_TEXTCNN(vocab_size=len(vocab), embed_dim=args.embed_dim, 
                                 num_classes=num_classes, ode_step=args.ode_step).to(device)
        elif args.model_type == 'node_textcnn_paral':
            model = NODE_TEXTCNN_PARALLEL(vocab_size=len(vocab), embed_dim=args.embed_dim, 
                                          num_classes=num_classes, ode_step=args.ode_step).to(device)
        
        # 词向量模型学习率通常大一点
        if args.lr < 1e-4: 
            print("Adjusting LR to 1e-3 for non-BERT model.")
            args.lr = 1e-3

    # B. BERT 类模型构建流程
    elif args.model_type in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        train_dataset = THUNewsDataset(train_data, tokenizer, 512)
        val_dataset = THUNewsDataset(val_data, tokenizer, 512)
        
        if args.fine_tune:
            mode_name = "finetune"
            current_batch_size = args.batch_size
            freeze_bert = False
        else:
            mode_name = "frozen"
            current_batch_size = args.batch_size * 2
            freeze_bert = True
            
        bert_base = BertModel.from_pretrained(args.bert_path)
        
        # 模型选择
        if args.model_type == 'bert':
            model = BertClassifier(bert_base, num_classes, freeze_bert=freeze_bert).to(device)
        elif args.model_type == 'textcnn':
            model = BERT_TextCNN(num_classes, bert_base, freeze_bert=freeze_bert).to(device)
        elif args.model_type == 'bert_node_textcnn':
            model = BERT_NODE_TEXTCNN(num_classes, bert_base, freeze_bert=freeze_bert, ode_step=args.ode_step).to(device)
        elif args.model_type == 'bert_node_textcnn_paral':
            model = BERT_NODE_TEXTCNN_PARALLEL(num_classes, bert_base, freeze_bert=freeze_bert, ode_step=args.ode_step).to(device)

    # --- 3. 训练循环 ---
    print(f"Start Training: Model={args.model_type}, Mode={mode_name}, Batch={current_batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=current_batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    save_path = os.path.join(args.output_dir, f"{args.model_type}_{mode_name}_best_f1.pth")
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Forward Pass 区分
            if args.model_type in VOCAB_MODELS:
                # 词向量模型只需要 input_ids
                outputs = model(inputs) 
            else:
                # BERT 模型需要 attention_mask
                mask = batch['attention_mask'].to(device)
                outputs = model(inputs, mask)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                if args.model_type in VOCAB_MODELS:
                    outputs = model(inputs)
                else:
                    mask = batch['attention_mask'].to(device)
                    outputs = model(inputs, mask)
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best F1! Saved.")
    
    print(f"Training Complete. Best Macro-F1: {best_f1:.6f}")
    print(f"MODEL_SAVED_AT: {save_path}")

if __name__ == "__main__":
    args = get_args()
    train(args)