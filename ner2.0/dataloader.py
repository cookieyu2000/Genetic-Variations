import os
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self, examples, tokenizer, label2id, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex['text']
        entities = ex['entities']

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        labels = [self.label2id['O']] * len(encoding['input_ids'])
        offset_mapping = encoding.pop('offset_mapping')

        # assign BIO labels
        for i_tok, (off_start, off_end) in enumerate(offset_mapping):
            for ent in entities:
                st = ent['start']
                ed = ent['end']
                lab = ent['label']
                # 跳過完全在實體之外的 tokens
                if off_end <= st or off_start >= ed:
                    continue
                prefix = 'B' if off_start <= st else 'I'
                key = f"{prefix}-{lab}"
                if key in self.label2id:
                    labels[i_tok] = self.label2id[key]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(labels)
        }

def parse_data(path, cache_path):
    # 如果已有 cache，直接讀取
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            examples = pickle.load(f)
        print(f"Loaded cached data from {cache_path}")
        return examples

    # 否則重新解析
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    examples = []
    i = 0
    pbar = tqdm(total=len(lines), desc='Parsing lines')
    while i < len(lines):
        pmid, tag, title = lines[i].split('|', 2)
        i += 1
        _, _, abstract = lines[i].split('|', 2)
        text = f"{title.strip()} {abstract.strip()}"
        i += 1
        entities = []
        # 正確判斷實體行：先 split 再 strip
        while i < len(lines) and '|' in lines[i]:
            parts = [p.strip() for p in lines[i].split('|')]
            if len(parts) >= 5 and parts[1].isdigit() and parts[2].isdigit():
                _, st, ed, word, lab = parts[:5]
                entities.append({'start': int(st), 'end': int(ed), 'label': lab})
                i += 1
                pbar.update(1)
            else:
                break
        examples.append({'text': text, 'entities': entities})
        pbar.update(1)
    pbar.close()

    # 緩存
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(examples, f)
    print(f"Cached parsed data to {cache_path}")
    return examples

def get_dataloaders(config):
    examples = parse_data(
        config['data']['dataset_path'],
        config['data']['cache_path']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    base_labels = config['labels']
    full_labels = ['O'] + [f"B-{l}" for l in base_labels] + [f"I-{l}" for l in base_labels]
    print("Full BIO labels:", full_labels)
    label2id = {lab: idx for idx, lab in enumerate(full_labels)}

    dataset = NERDataset(examples, tokenizer, label2id, config['model']['max_len'])
    total = len(dataset)
    train_len = int(total * config['data']['train_split'])
    valid_len = int(total * config['data']['valid_split'])
    test_len = total - train_len - valid_len
    gen = torch.Generator().manual_seed(config['training']['seed'])
    train_ds, valid_ds, test_ds = random_split(
        dataset, [train_len, valid_len, test_len], generator=gen
    )

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config['training']['batch_size'])
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'])
    return train_loader, valid_loader, test_loader, full_labels
