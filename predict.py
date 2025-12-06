import argparse
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from model_utils import (setup_seed, jsonl_to_dataframe, THUNewsDataset, TraditionalDataset,
                         BertClassifier, TextCNN, NODE_TEXTCNN, TraditionalTextCNN, Vocab)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, help="传统模型需要的词表路径")
    parser.add_argument("--fine_tune", action="store_true")
    
    parser.add_argument("--data_path", type=str, default="../data/paper_full_data.jsonl")
    parser.add_argument("--bert_path", type=str, default="/root/lanyun-fs/models/Bert")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--embed_dim", type=int, default=512) 
    parser.add_argument("--ode_step", type=int, default=11)
    return parser.parse_args()

def predict(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_bert = not args.fine_tune
    
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
    # =============================================================
    
    # 2. 加载模型与数据
    if args.model_type == 'traditional':
        # 自动推断 vocab 路径 (如果在 model_path 同级)
        if not args.vocab_path:
            model_dir = os.path.dirname(args.model_path)
            possible_path = os.path.join(model_dir, "traditional_vocab.json")
            if os.path.exists(possible_path):
                args.vocab_path = possible_path
            
        print(f"Loading Vocab from {args.vocab_path}")
        vocab = Vocab()
        vocab.load(args.vocab_path)
        
        test_dataset = TraditionalDataset(test_data, vocab, 512)
        model = TraditionalTextCNN(len(vocab), args.embed_dim, len(labels)).to(device)
        
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        test_dataset = THUNewsDataset(test_data, tokenizer, 512)
        
        bert_base = BertModel.from_pretrained(args.bert_path)
        if args.model_type == 'bert':
            model = BertClassifier(bert_base, len(labels), freeze_bert=freeze_bert).to(device)
        elif args.model_type == 'textcnn':
            model = TextCNN(len(labels), bert_base, freeze_bert=freeze_bert).to(device)
        elif args.model_type == 'node':
            model = NODE_TEXTCNN(len(labels), bert_base, freeze_bert=freeze_bert, ode_step=args.ode_step).to(device)
            
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. 加载权重
    print(f"Loading weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # 4. 预测
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if args.model_type == 'traditional':
                inputs = batch['input_ids'].to(device)
                outputs = model(inputs)
            else:
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                outputs = model(inputs, mask)
                
            labels = batch['label'].to(device)
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