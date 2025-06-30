# FILE: ner2.0/evaluate_confusion.py

import os
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification
# 關鍵：從子模組直接匯入，絕對不要再用「from seqeval.metrics import …」
from seqeval.metrics.sequence_labeling import classification_report, f1_score
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from dataloader import get_dataloaders

def load_config(path='ner2.0/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model(output_dir):
    model = AutoModelForTokenClassification.from_pretrained(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

def evaluate():
    # 1. 載入 config 及 test loader
    cfg = load_config()
    _, _, test_loader, full_labels = get_dataloaders(cfg)

    # 2. 載入訓練好的模型
    model, device = load_model(cfg['training']['output_dir'])
    model.eval()

    seq_preds, seq_labels = [], []
    token_preds, token_labels, token_probs = [], [], []

    # 3. 遍歷測試集
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        with torch.no_grad():
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits
            # token-level 機率
            probs     = F.softmax(logits, dim=-1).cpu().numpy()
            preds_idx = probs.argmax(axis=-1)
            labels_idx= labels.cpu().numpy()
            mask      = attention_mask.cpu().numpy() == 1

            # 收集每個 token 的預測與真實標籤
            for p_row, l_row, prob_row, m_row in zip(preds_idx, labels_idx, probs, mask):
                sp, sl = [], []
                for p_i, l_i, prob_i, m_i in zip(p_row, l_row, prob_row, m_row):
                    if m_i:
                        sp.append(full_labels[p_i])
                        sl.append(full_labels[l_i])
                        token_preds.append(p_i)
                        token_labels.append(l_i)
                        token_probs.append(prob_i)
                seq_preds.append(sp)
                seq_labels.append(sl)

    # 4. 計算 token-level 指標
    acc  = accuracy_score(token_labels, token_preds)
    prec = precision_score(token_labels, token_preds, average='macro', zero_division=0)
    # multi-class AUC
    y_true_bin = label_binarize(token_labels, classes=list(range(len(full_labels))))
    try:
        auc = roc_auc_score(y_true_bin, token_probs, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')

    # 5. 計算 sequence-level F1（entity-level）
    f1 = f1_score(seq_labels, seq_preds)

    # 6. 輸出 token-level 數值（4 位小數）
    print(f"Test → accuracy: {acc:.4f}, precision: {prec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # 7. 輸出每個類別的二元混淆矩陣
    print("\nConfusion Matrix per class:")
    for idx, label in enumerate(full_labels):
        y_true_binary = [1 if t == idx else 0 for t in token_labels]
        y_pred_binary = [1 if p == idx else 0 for p in token_preds]
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        print(f"\nLabel: {label}")
        print(cm)

    # 8. 輸出 sequence-level classification report（4 位小數）
    if seq_labels and any(lbl != 'O' for seq in seq_labels for lbl in seq):
        print("\nSequence-level Classification Report:")
        print(classification_report(seq_labels, seq_preds, digits=4))
    else:
        print("No non-O entity labels; skipped classification report.")

if __name__ == '__main__':
    evaluate()
