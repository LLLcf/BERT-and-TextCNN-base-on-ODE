import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchdiffeq import odeint
import pandas as pd
import json
import numpy as np
import random
import os
import re
import jieba
from collections import Counter
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# --- 1. 全局基础工具 ---
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)
    return pd.DataFrame(data)

# --- 2. 文本处理与特征工程工具 (For Sklearn Models) ---

# 停用词 (示例)
STOPWORDS = set([
    '\n', '\t', ' ', '　', ',', '.', '!', '?', ';', ':', '、', '，', '。', '！', '？',
    '的', '了', '是', '在', '和', '有', '就', '都', '而', '及', '与', '这', '那'
])

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

class BertVectorizor:
    """从本地路径加载 BERT 并提取特征 (用于 Sklearn)"""
    def __init__(self, model_path, batch_size=32, device=None):
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading BERT for Feature Extraction: {model_path} (Device: {self.device})")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading BERT: {e}")
            raise e

    def transform(self, texts):
        print(f"Extracting BERT Features ({len(texts)} samples)...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT Embedding"):
            batch_texts = texts[i : i + self.batch_size]
            batch_texts = [clean_text(t) for t in batch_texts] 
            
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

# --- 3. 词表工具 (For Traditional DL Model) ---
class Vocab:
    def __init__(self, tokens=None, min_freq=1, specials=['<pad>', '<unk>']):
        self.specials = specials
        self.token2id = {token: idx for idx, token in enumerate(specials)}
        self.id2token = {idx: token for idx, token in enumerate(specials)}
        
        if tokens:
            self.build_vocab(tokens, min_freq)

    def build_vocab(self, tokens, min_freq):
        counter = Counter(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
                
    def __len__(self):
        return len(self.token2id)

    def __getitem__(self, token):
        return self.token2id.get(token, self.token2id['<unk>'])

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False)
            
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.token2id = json.load(f)
            self.id2token = {v: k for k, v in self.token2id.items()}

# --- 4. 数据集类 ---

# 1. BERT 系列数据集
class THUNewsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, 'content']
        label = self.dataset.loc[idx, 'label']
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 2. 传统模型数据集 (Jieba + Vocab)
class TraditionalDataset(Dataset):
    def __init__(self, dataset, vocab, max_length):
        self.dataset = dataset
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, 'content']
        label = self.dataset.loc[idx, 'label']
        
        # 分词
        tokens = jieba.lcut(text)
        # 截断与填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        token_ids = [self.vocab[token] for token in tokens]
        
        # Padding (用 0 填充，假设 <pad> id 为 0)
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [0] * padding_length
            
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

# --- 5. 辅助模块 ---
class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.net.weight, mean=0, std=0.1)
        nn.init.constant_(self.net.bias, 0)
    def forward(self, t, x):
        return self.net(x)

# --- 6. 模型定义 ---

class TraditionalTextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3, 4, 5], num_channels=[256, 256, 256]):
        super(TraditionalTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cnn_layers = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=c, kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        self.pool = GlobalMaxPool1d()
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2), 
            nn.Linear(sum(num_channels), num_classes)
        )

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        x = embed.permute(0, 2, 1)
        y = []
        for layer in self.cnn_layers:
            out = layer(x)
            out = self.pool(out).squeeze(-1)
            y.append(out)
        y = torch.cat(y, dim=1)
        return self.classify(y)

class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, freeze_bert=True, dropout=0.2):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        if freeze_bert:
            for param in self.bert.parameters(): param.requires_grad = False
        else:
            for param in self.bert.parameters(): param.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, mask):
        output = self.bert(input_ids=input_ids, attention_mask=mask)
        dropout_output = self.dropout(output[1])
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)

class TextCNN(nn.Module):
    def __init__(self, num_classes, bert_model, freeze_bert=True, kernel_sizes=[3, 4, 5], num_channels=[256, 256, 256]):
        super(TextCNN, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters(): param.requires_grad = not freeze_bert
        self.embedding_dim = self.bert.config.hidden_size
        self.cnn_layers = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=self.embedding_dim, out_channels=c, kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        self.pool = GlobalMaxPool1d()
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(num_channels), num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        inputs = outputs.last_hidden_state.permute(0, 2, 1) 
        y = []
        for layer in self.cnn_layers:
            x = layer(inputs)
            x = self.pool(x).squeeze(-1)
            y.append(x)
        y = torch.cat(y, dim=1)
        return self.classify(y)

class NODE_TEXTCNN(nn.Module):
    def __init__(self, num_classes, bert_model, freeze_bert=True, kernel_sizes=[3, 4, 5], num_channels=[128, 128, 128], 
                 ode_hidden=768, ode_step=11, n_layers=1):
        super(NODE_TEXTCNN, self).__init__()
        self.bert = bert_model
        for param in self.bert.parameters(): param.requires_grad = not freeze_bert
        self.ode_func = ODEFunc(ode_hidden)
        self.ode_step = ode_step
        self.n_layers = n_layers
        self.cnn_layers = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels=ode_hidden, out_channels=c, kernel_size=k),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            )
            self.cnn_layers.append(cnn)
        self.pool = GlobalMaxPool1d()
        self.classify = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(sum(num_channels) * n_layers, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        curr_x = outputs.last_hidden_state 
        all_ode_outputs = []
        for _ in range(self.n_layers):
            batch_size, seq_len, hidden_dim = curr_x.shape
            node_input = curr_x.reshape(-1, hidden_dim) 
            t = torch.linspace(0., 1., steps=self.ode_step).to(curr_x.device)
            node_output = odeint(self.ode_func, node_input, t, rtol=1e-3, atol=1e-3)[-1]
            node_output = node_output.reshape(batch_size, seq_len, hidden_dim).permute(0, 2, 1) 
            all_ode_outputs.append(node_output)
            curr_x = node_output.permute(0, 2, 1)
        all_pooled = []
        for ode_out in all_ode_outputs:
            for layer in self.cnn_layers:
                conv_out = layer(ode_out)
                pooled = self.pool(conv_out).squeeze(-1)
                all_pooled.append(pooled)
        logits = self.classify(torch.cat(all_pooled, dim=1))
        return logits