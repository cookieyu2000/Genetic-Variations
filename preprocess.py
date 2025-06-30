# preprocess.py
from util import *
import os
import json
import yaml
import pickle
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(raw_data_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    labels = []
    citation_ids = []
    for entry in data:
        texts.append(entry['input'])  # 不進行 ' '.join
        labels.append(entry['output'])
        citation_ids.append(entry['citation_id'])
    return texts, labels, citation_ids

def preprocess_data(texts, labels):
    label2id = {label: idx for idx, label in enumerate(sorted(list(set(labels))))}
    id2label = {idx: label for label, idx in label2id.items()}
    numeric_labels = [label2id[label] for label in labels]
    return texts, numeric_labels, label2id, id2label

def split_data(texts, labels, citation_ids, config, seed):
    train_size = config['data']['train_split']
    valid_size = config['data']['valid_split']
    test_size = config['data']['test_split']
    
    X_train, X_temp, y_train, y_temp, cid_train, cid_temp = train_test_split(
        texts, labels, citation_ids,
        train_size=train_size,
        random_state=seed,
        stratify=labels
    )
    
    valid_relative_size = valid_size / (valid_size + test_size)
    X_valid, X_test, y_valid, y_test, cid_valid, cid_test = train_test_split(
        X_temp, y_temp, cid_temp,
        train_size=valid_relative_size,
        random_state=seed,
        stratify=y_temp
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, cid_train, cid_valid, cid_test

def upsample_training_data(X_train, y_train, cid_train, seed):
    df_train = pd.DataFrame({'text': X_train, 'label': y_train, 'citation_id': cid_train})
    class_counts = df_train['label'].value_counts()
    max_count = class_counts.max()
    
    df_upsampled = df_train.copy()
    for label, count in class_counts.items():
        if count < max_count:
            df_minority = df_train[df_train['label'] == label]
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=max_count - count,
                random_state=seed
            )
            df_upsampled = pd.concat([df_upsampled, df_minority_upsampled])
    
    df_upsampled = df_upsampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df_upsampled['text'].tolist(), df_upsampled['label'].tolist(), df_upsampled['citation_id'].tolist()

def save_split_data(datasets, config, label2id, id2label):
    split_data_dir = config['paths']['split_data_dir']
    os.makedirs(split_data_dir, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        split_json_path = os.path.join(split_data_dir, f"{split}.json")
        split_pkl_path = os.path.join(split_data_dir, f"{split}.pkl")
        
        data_for_json = []
        for i in range(len(datasets[split]['texts'])):
            data_for_json.append({
                'citation_id': datasets[split]['citation_ids'][i],
                'input': datasets[split]['texts'][i],
                'output': id2label[datasets[split]['labels'][i]]
            })

        with open(split_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_for_json, f, ensure_ascii=False, indent=4)
        
        pickle_data = {
            'texts': datasets[split]['texts'],
            'labels': datasets[split]['labels']
        }
        with open(split_pkl_path, 'wb') as f:
            pickle.dump(pickle_data, f)
    
    label2id_pkl_path = os.path.join(split_data_dir, 'label2id.pkl')
    id2label_pkl_path = os.path.join(split_data_dir, 'id2label.pkl')
    
    with open(label2id_pkl_path, 'wb') as f:
        pickle.dump(label2id, f)
    
    with open(id2label_pkl_path, 'wb') as f:
        pickle.dump(id2label, f)

def main():
    config = load_config()
    texts, labels, citation_ids = load_data(config['data']['raw_data_path'])
    texts, labels, label2id, id2label = preprocess_data(texts, labels)
    
    X_train, X_valid, X_test, y_train, y_valid, y_test, cid_train, cid_valid, cid_test = split_data(
        texts, labels, citation_ids, config, config['random_seed']
    )
    
    X_train_upsampled, y_train_upsampled, cid_train_upsampled = upsample_training_data(
        X_train, y_train, cid_train, config['random_seed']
    )
    
    datasets = {
        'train': {
            'texts': X_train_upsampled,
            'labels': y_train_upsampled,
            'citation_ids': cid_train_upsampled
        },
        'valid': {
            'texts': X_valid,
            'labels': y_valid,
            'citation_ids': cid_valid
        },
        'test': {
            'texts': X_test,
            'labels': y_test,
            'citation_ids': cid_test
        }
    }
    
    save_split_data(datasets, config, label2id, id2label)

if __name__ == '__main__':
    main()
