import os
import json
import yaml
import torch
import torch.nn.functional as F
from transformers import AdamW, AutoTokenizer
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataloader import get_dataloaders, parse_data, NERDataset
from model import NERModel

def load_config(path='ner2.0/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train():
    cfg = load_config()
    torch.manual_seed(cfg['training']['seed'])

    # Early stopping params
    patience = cfg['training'].get('patience', 3)
    epochs_no_improve = 0
    best_f1 = 0.0

    # Sample Inspection & Entity Check
    examples = parse_data(cfg['data']['dataset_path'], cfg['data']['cache_path'])
    print("第一筆 example 的 entities:", examples[0]['entities'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['model_name'])
    base_labels = cfg['labels']
    full_labels = ['O'] + [f"B-{l}" for l in base_labels] + [f"I-{l}" for l in base_labels]
    label2id = {lab: idx for idx, lab in enumerate(full_labels)}

    ds = NERDataset(examples, tokenizer, label2id, cfg['model']['max_len'])
    sm = ds[0]
    toks = tokenizer.convert_ids_to_tokens(sm['input_ids'].tolist())
    labs = [full_labels[i] for i in sm['labels'].tolist()]
    print("\n=== Sample Inspection ===")
    print("Text:\n", examples[0]['text'])
    print("\nTokens:\n", toks)
    print("\nLabels:\n", labs)
    print("=== End Sample Inspection ===\n")

    # Prepare DataLoaders & Model
    train_loader, valid_loader, test_loader, full_labels = get_dataloaders(cfg)
    model = NERModel(cfg['model']['model_name'], len(full_labels))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']))

    total_epochs = cfg['training']['epochs']

    # Training & Validation Loop with Early Stopping
    for epoch in range(1, total_epochs + 1):
        # -- Train --
        model.train()
        total_loss = 0.0
        train_pbar = tqdm(
            train_loader,
            desc=f"[Train] Epoch {epoch}/{total_epochs}",
            leave=False
        )
        for batch in train_pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch}/{total_epochs} - Train loss: {total_loss/len(train_loader):.4f}")

        # -- Validate --
        model.eval()
        seq_preds, seq_labels = [], []
        tok_preds, tok_labels, tok_probs = [], [], []
        valid_pbar = tqdm(
            valid_loader,
            desc=f"[Valid] Epoch {epoch}/{total_epochs}",
            leave=False
        )
        for batch in valid_pbar:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds_idx = probs.argmax(axis=-1)
                labels_idx = labels.cpu().numpy()
                mask = attention_mask.cpu().numpy() == 1
                for p_row, l_row, prob_row, m_row in zip(
                    preds_idx, labels_idx, probs, mask
                ):
                    seq_p, seq_l = [], []
                    for p, l, prob, m in zip(p_row, l_row, prob_row, m_row):
                        if m:
                            seq_p.append(full_labels[p])
                            seq_l.append(full_labels[l])
                            tok_preds.append(p)
                            tok_labels.append(l)
                            tok_probs.append(prob)
                    seq_preds.append(seq_p)
                    seq_labels.append(seq_l)

        # Compute metrics
        acc   = accuracy_score(tok_labels, tok_preds)
        prec  = precision_score(tok_labels, tok_preds, average='macro', zero_division=0)
        y_bin = label_binarize(tok_labels, classes=list(range(len(full_labels))))
        try:
            auc = roc_auc_score(y_bin, tok_probs, average='macro', multi_class='ovr')
        except:
            auc = float('nan')
        val_f1 = f1_score(seq_labels, seq_preds)

        print(
            f"Validation → acc: {acc:.4f}, prec: {prec:.4f}, "
            f"F1: {val_f1:.4f}, AUC: {auc:.4f}"
        )
        if any(lbl != 'O' for seq in seq_labels for lbl in seq):
            print(classification_report(seq_labels, seq_preds))
        else:
            print("No non-O entity labels; skipped report.")

        # -- Early Stopping & Save Best --
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            metrics = {'accuracy': acc, 'precision': prec, 'f1': val_f1, 'auc': auc}
            os.makedirs(cfg['training']['output_dir'], exist_ok=True)

            # 1) 覆盖 config 中的 id2label 和 label2id
            # full_labels 就是 ['O', 'B-gene', 'I-gene', ..., 'I-genomicregion']
            model.bert.config.id2label = {i: lab for i, lab in enumerate(full_labels)}
            model.bert.config.label2id = {lab: i for i, lab in enumerate(full_labels)}

            # 2) 以 Huggingface 格式保存模型（包含覆盖后的 config.json）
            model.bert.save_pretrained(cfg['training']['output_dir'])

            # 3) 同步保存 tokenizer
            tokenizer.save_pretrained(cfg['training']['output_dir'])

            # 4) 额外保存一个 PyTorch state_dict
            torch.save(
                model.bert.state_dict(),
                os.path.join(cfg['training']['output_dir'], 'best_ner.pt')
            )

            # 5) 保存度量指标
            with open(os.path.join(cfg['training']['output_dir'], 'best_metrics.json'), 'w') as mf:
                json.dump(metrics, mf, indent=2)

            print(
                f"Saved best model & tokenizer & weights & metrics "
                f"(F1={best_f1:.4f}) to {cfg['training']['output_dir']}\n"
            )
        else:
            epochs_no_improve += 1
            print(f"No F1 improvement for {epochs_no_improve}/{patience} epochs.\n")
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered. "
                    f"Stopping after epoch {epoch}.\n"
                )
                break

if __name__ == '__main__':
    train()
