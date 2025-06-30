# util.py

import matplotlib
matplotlib.use('Agg')  # 使用無頭後端，避免 GUI 錯誤
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import pickle

def setup_logging(log_file, level=logging.INFO):
    """
    配置日誌系統，將日誌輸出到文件和控制台。
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=level
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def set_seed(seed):
    """
    設定隨機種子以確保可重現性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 確保 cudnn 的行為是可重現的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")
    
def train_epoch(model, data_loader, optimizer, device, criterion, scaler, debug_batches=2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 建立一個 tqdm 進度條，desc 可以自訂
    progress_bar = tqdm(data_loader, desc="Training", leave=True, dynamic_ncols=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        # ========= DEBUG 輸出: 僅前 debug_batches 個 batch =========
        # if batch_idx < debug_batches:
        #     print(f"\n[DEBUG] --- Batch {batch_idx} info ---")
        #     print(f"input_ids.shape: {input_ids.shape}")
        #     print(f"attention_mask.shape: {attention_mask.shape}")
        #     print(f"labels.shape: {labels.shape}")

        #     # 列印前 5 個 token, label
        #     print("input_ids[0][:5]:", input_ids[0][:5].detach().cpu().tolist() if input_ids.dim() > 1 else input_ids[0].detach().cpu().tolist())
        #     print("attention_mask[0][:5]:", attention_mask[0][:5].detach().cpu().tolist() if attention_mask.dim() > 1 else attention_mask[0].detach().cpu().tolist())
            
        #     if labels.dim() == 0:
        #         print("labels item:", labels.item())
        #     else:
        #         print("labels[:5]:", labels[:5].detach().cpu().tolist())
                
        # ========= forward =========
        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 動態更新 tqdm 的後綴資訊
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = correct / total
        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{accuracy:.4f}"
        })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return accuracy, avg_loss


def eval_model(model, data_loader, device, loss_fn, num_labels):
    """
    評估模型在給定的 DataLoader 上的表現。

    Args:
        model: 已載入的分類模型。
        data_loader: 測試資料的 DataLoader。
        device: 計算設備（CPU 或 GPU）。
        loss_fn: 損失函數。
        num_labels: 類別數量。

    Returns:
        tuple: 包含損失、準確率、混淆矩陣和各種評估指標的元組。
    """
    model.eval()
    losses = []
    correct_predictions = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    loss = np.mean(losses)
    acc = correct_predictions.double() / len(data_loader.dataset)
    auc = roc_auc_score(all_labels, [prob[1] for prob in all_probs])

    
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(
        all_labels,
        all_preds,
        average='weighted',
        zero_division=0
    )
    recall = recall_score(
        all_labels,
        all_preds,
        average='binary',
        zero_division=0,
        pos_label=1
    )
    f1 = f1_score(
        all_labels,
        all_preds,
        average='weighted',
        zero_division=0
    )
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return acc, loss, cm, metrics

def plot_confusion_matrix(cm, classes, output_path):
    """
    繪製並保存混淆矩陣。
    
    Args:
        cm: 混淆矩陣。
        classes: 類別名稱列表。
        output_path: 圖像保存路徑。
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def log_model_info(model):
    """
    記錄模型的架構和參數量。
    """
    # 記錄模型架構
    logging.info("Model Architecture:")
    logging.info(model)
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    logging.info(f"Total Parameters: {total_params}")
    logging.info(f"Trainable Parameters: {trainable_params}")
    logging.info(f"Non-trainable Parameters: {non_trainable_params}")

class EarlyStopping:
    """
    早停機制，用於在驗證指標不再改善時提前停止訓練。
    
    Args:
        patience (int): 等待多少個 epoch 指標沒有改善後停止訓練。
        verbose (bool): 是否輸出詳細信息。
        delta (float): 指標改善的最小變化量。
        mode (str): 'min' 或 'max'，決定指標是要最小化還是最大化。
    """
    def __init__(self, patience=5, verbose=False, delta=0, mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.best_metric = None

    def __call__(self, metric, model, save_path):
        score = metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif self.mode == 'min':
            if score < self.best_score - self.delta:
                self.best_score = score
                self.save_checkpoint(model, save_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        elif self.mode == 'max':
            if score > self.best_score + self.delta:
                self.best_score = score
                self.save_checkpoint(model, save_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model, save_path):
        """
        保存模型的最佳狀態。
        """
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            logging.info(f'Validation score improved. Saving model to {save_path}')

    def reset_counter(self):
        """
        重置早停計數器。
        """
        self.counter = 0
        if self.verbose:
            logging.info("EarlyStopping counter reset due to learning rate adjustment.")
