import argparse
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 引入自定义模块 (确保包含所有新模型类)
from model_utils import (
    setup_seed, jsonl_to_dataframe, 
    THUNewsDataset, TraditionalDataset, Vocab,
    # 模型类
    BertClassifier, BERT_TextCNN, TraditionalTextCNN,
    NODE_TEXTCNN, NODE_TEXTCNN_PARALLEL,
    BERT_NODE_TEXTCNN, BERT_NODE_TEXTCNN_PARALLEL
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型 .pth 文件路径")
    parser.add_argument("--model_type", type=str, required=True, 
                        choices=[
                            'traditional', 'node_textcnn', 'node_textcnn_paral', # 词表类
                            'bert', 'textcnn', 'bert_node_textcnn', 'bert_node_textcnn_paral' # BERT类
                        ])
    parser.add_argument("--vocab_path", type=str, help="传统模型需要的词表路径 (如果不指定则尝试自动推断)")
    parser.add_argument("--fine_tune", action="store_true", help="如果是 BERT 模型，是否是微调模式 (影响参数加载)")
    
    parser.add_argument("--data_path", type=str, default="../data/paper_full_data.jsonl")
    parser.add_argument("--bert_path", type=str, default="/root/lanyun-fs/models/Bert")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--embed_dim", type=int, default=512, help="词表类模型的词向量维度") 
    parser.add_argument("--ode_step", type=int, default=11)
    return parser.parse_args()

def predict(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 定义模型分组
    VOCAB_MODELS = ['traditional', 'node_textcnn', 'node_textcnn_paral']
    BERT_MODELS = ['bert', 'textcnn', 'bert_node_textcnn', 'bert_node_textcnn_paral']

    # 1. 恢复与训练完全一致的数据划分
    # =============================================================
    df = jsonl_to_dataframe(args.data_path)
    # 下采样 (必须与训练一致)
    df = df.groupby('类别', group_keys=False).apply(
        lambda x: x.sample(frac=1/5, random_state=args.seed)
    ).reset_index(drop=True)
    
    labels = sorted(list(set(df['类别'])))
    label2id = {cat: ids for ids, cat in enumerate(labels)}
    id2label = {ids: cat for cat, ids in label2id.items()}
    df['label'] = df['类别'].apply(lambda x: label2id[str(x)])
    
    # 划分 (Train/Val/Test = 8:1:1) - 仅取 Test
    df = df.reset_index(drop=True)
    train_data = df.sample(frac=0.8, random_state=args.seed)
    remaining_data = df.drop(train_data.index)
    val_data = remaining_data.sample(frac=0.5, random_state=args.seed)
    test_data = remaining_data.drop(val_data.index).reset_index(drop=True)
    
    print(f"Test Dataset Size: {len(test_data)}")
    num_classes = len(labels)
    # =============================================================
    
    # 2. 模型与数据加载
    
    # === 分支 A: 词表类模型 (Jieba + Embedding) ===
    if args.model_type in VOCAB_MODELS:
        # 自动推断 vocab 路径
        if not args.vocab_path:
            model_dir = os.path.dirname(args.model_path)
            # 尝试推断可能的命名: traditional_vocab.json 或 node_textcnn_vocab.json
            possible_names = ["traditional_vocab.json", f"{args.model_type}_vocab.json"]
            for name in possible_names:
                p = os.path.join(model_dir, name)
                if os.path.exists(p):
                    args.vocab_path = p
                    break

        if not args.vocab_path or not os.path.exists(args.vocab_path):
            raise FileNotFoundError(f"Vocab file not found. Please specify --vocab_path. (Tried in {os.path.dirname(args.model_path)})")
            
        print(f"Loading Vocab from {args.vocab_path}")
        vocab = Vocab()
        vocab.load(args.vocab_path)
        
        test_dataset = TraditionalDataset(test_data, vocab, 512)
        
        # 初始化模型架构
        if args.model_type == 'traditional':
            model = TraditionalTextCNN(len(vocab), args.embed_dim, num_classes).to(device)
        elif args.model_type == 'node_textcnn':
            model = NODE_TEXTCNN(len(vocab), args.embed_dim, num_classes, ode_step=args.ode_step).to(device)
        elif args.model_type == 'node_textcnn_paral':
            model = NODE_TEXTCNN_PARALLEL(len(vocab), args.embed_dim, num_classes, ode_step=args.ode_step).to(device)

    # === 分支 B: BERT 类模型 ===
    elif args.model_type in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        test_dataset = THUNewsDataset(test_data, tokenizer, 512)
        
        bert_base = BertModel.from_pretrained(args.bert_path)
        # 预测时通常不需要梯度，freeze_bert 参数主要影响 requires_grad，对 load_state_dict 无影响
        # 但为了保持一致性，还是设置一下
        freeze_bert = not args.fine_tune
        
        # 初始化模型架构
        if args.model_type == 'bert':
            model = BertClassifier(bert_base, num_classes, freeze_bert=freeze_bert).to(device)
        elif args.model_type == 'textcnn':
            model = BERT_TextCNN(num_classes, bert_base, freeze_bert=freeze_bert).to(device)
        elif args.model_type == 'bert_node_textcnn':
            model = BERT_NODE_TEXTCNN(num_classes, bert_base, freeze_bert=freeze_bert, ode_step=args.ode_step).to(device)
        elif args.model_type == 'bert_node_textcnn_paral':
            model = BERT_NODE_TEXTCNN_PARALLEL(num_classes, bert_base, freeze_bert=freeze_bert, ode_step=args.ode_step).to(device)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. 加载权重
    print(f"Loading weights from {args.model_path}")
    # 注意：strict=False 有时用于微调后加载 (如 BERT 部分参数差异)，但通常应该用 True 确保完整匹配
    # 如果遇到 BERT pooler 层的 key mismatch，可以考虑 strict=False
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
    model.eval()

    # 4. 预测
    all_preds, all_labels = [], []
    print("Start Prediction...")

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            if args.model_type in VOCAB_MODELS:
                outputs = model(inputs)
            else:
                mask = batch['attention_mask'].to(device)
                outputs = model(inputs, mask)
                
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 5. 输出结果
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("-" * 30)
    print(f"Model: {args.model_type}")
    print(f"Type: {'Fine-tune' if args.fine_tune else 'Frozen/Scratch'}")
    print(f"Test Accuracy: {acc:.6f}")
    print(f"Test Macro F1: {f1:.6f}") 
    print("-" * 30)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(label2id))], digits=6))

if __name__ == "__main__":
    args = get_args()
    predict(args)