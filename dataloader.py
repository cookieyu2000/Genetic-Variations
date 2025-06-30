# dataloader.py
import json
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
import torch
import os
import pickle

# max_samples = 100

class CustomDataset(Dataset):
    def __init__(self, file_path, label2id, tokenizer, max_length, max_paragraphs, stride=128, sliding_window=True):
        """
        初始化數據集，將文章分段處理，根據 `sliding_window` 動態啟用滑動窗口功能。
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.max_paragraphs = max_paragraphs
        self.stride = stride
        self.sliding_window = sliding_window
        self.samples = []

        for item in data:
            paragraphs = item['input']
            label = label2id[item['output']]
            
            processed_paragraphs = []
            for p in paragraphs:
                tokens = self.tokenizer.tokenize(p)
                window_size = max_length - 2

                if self.sliding_window:
                    # 使用滑動窗口處理段落
                    for i in range(0, len(tokens), stride):
                        sub_tokens = tokens[i:i + window_size]
                        if not sub_tokens:
                            continue
                        sub_p = self.tokenizer.convert_tokens_to_string(sub_tokens)
                        processed_paragraphs.append(sub_p)
                        if len(processed_paragraphs) >= max_paragraphs:
                            break
                else:
                    # 無滑動窗口，直接處理整個段落
                    sub_p = self.tokenizer.convert_tokens_to_string(tokens[:window_size])
                    processed_paragraphs.append(sub_p)

                if len(processed_paragraphs) >= max_paragraphs:
                    break
            
            # 截斷或填補段落數量
            processed_paragraphs = processed_paragraphs[:max_paragraphs]
            processed_paragraphs += [""] * (max_paragraphs - len(processed_paragraphs))
            
            # Tokenize 每個段落
            input_ids_list, attention_mask_list = [], []
            for p in processed_paragraphs:
                encoding = self.tokenizer(
                    p,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                input_ids_list.append(encoding['input_ids'].squeeze(0))
                attention_mask_list.append(encoding['attention_mask'].squeeze(0))
            
            input_ids_tensor = torch.stack(input_ids_list, dim=0)
            attention_mask_tensor = torch.stack(attention_mask_list, dim=0)
            self.samples.append((input_ids_tensor, attention_mask_tensor, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.samples[idx]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(config):
    tokenizer = AutoTokenizer.from_pretrained(config['data']['tokenizer_name'])

    with open(os.path.join(config['paths']['split_data_dir'], 'label2id.pkl'), 'rb') as f:
        label2id = pickle.load(f)
    with open(os.path.join(config['paths']['split_data_dir'], 'id2label.pkl'), 'rb') as f:
        id2label = pickle.load(f)

    train_dataset = CustomDataset(
        file_path=os.path.join(config['paths']['split_data_dir'], "train.json"),
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        max_paragraphs=config['data']['max_paragraphs'],
        stride=config['data'].get('stride', 128),
        sliding_window=config['data'].get('sliding_window', True)  # 從 config 動態控制
    )

    valid_dataset = CustomDataset(
        file_path=os.path.join(config['paths']['split_data_dir'], "valid.json"),
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        max_paragraphs=config['data']['max_paragraphs'],
        stride=config['data'].get('stride', 128),
        sliding_window=config['data'].get('sliding_window', True)  # 從 config 動態控制
    )

    test_dataset = CustomDataset(
        file_path=os.path.join(config['paths']['split_data_dir'], "test.json"),
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=config['data']['max_length'],
        max_paragraphs=config['data']['max_paragraphs'],
        stride=config['data'].get('stride', 128),
        sliding_window=config['data'].get('sliding_window', True)  # 從 config 動態控制
    )
    
    # # 如果您想只測試少量資料，可以設定 max_samples (例如 100)
    # if max_samples is not None and max_samples > 0:
    #     train_subset = Subset(train_dataset, list(range(min(len(train_dataset), max_samples))))
    #     valid_subset = Subset(valid_dataset, list(range(min(len(valid_dataset), max_samples))))
    #     test_subset  = Subset(test_dataset,  list(range(min(len(test_dataset),  max_samples))))
    # else:
    #     train_subset = train_dataset
    #     valid_subset = valid_dataset
    #     test_subset  = test_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    return train_loader, valid_loader, test_loader, label2id, id2label
