# train.py

import os
import yaml
import logging
import importlib
import inspect
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.optim import AdamW

from dataloader import create_data_loaders
from util import (
    EarlyStopping,
    setup_logging,
    plot_confusion_matrix,
    train_epoch,
    eval_model,
    set_seed
)

import warnings
warnings.filterwarnings("ignore")


def log_model_architecture_and_params(model):
    logging.info("Model architecture:\n" + str(model))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters:   {total}")
    logging.info(f"Trainable params:   {trainable}")


def log_loader_statistics(loaders, id2label):
    for split, loader in loaders.items():
        labels = []
        for batch in loader:
            labels.extend(batch["labels"].tolist())
        dist = Counter(labels)
        dist = {id2label[k]: v for k, v in dist.items()}
        logging.info(f"{split.capitalize()} set: {len(labels)} samples, distribution: {dist}")


def main():
    # argparse for config path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # 1) load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2) prepare output dirs
    logging_name = config["data"]["logging_name"]
    folder = os.path.splitext(logging_name)[0]
    out_dir = os.path.join(config["paths"]["base_output_dir"], folder)
    os.makedirs(out_dir, exist_ok=True)
    config["paths"]["timestamped_dir"] = out_dir
    config["paths"]["best_model_path"] = os.path.join(out_dir, "best_model.pt")

    # 3) setup logging
    log_file = os.path.join(out_dir, logging_name)
    setup_logging(log_file, level=getattr(logging, config["logging"]["level"]))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    logging.info(f"Configuration saved to {out_dir}")

    # 4) set seed
    set_seed(config["random_seed"])

    # 5) data loaders
    logging.info("Loading data...")
    train_loader, valid_loader, test_loader, label2id, id2label = create_data_loaders(config)
    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    log_loader_statistics(loaders, id2label)

    # 6) device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 7) dynamic model instantiation
    model_type = config["model"]["type"]
    model_module = importlib.import_module("model")
    if not hasattr(model_module, model_type):
        raise ValueError(f"Model class '{model_type}' not found in model.py")
    ModelClass = getattr(model_module, model_type)

    params = config["model"].copy()
    if "pretrained_model" in params:
        params["pretrained_model_name_or_path"] = params.pop("pretrained_model")
    if "transformer" in config:
        params["transformer_config"] = config["transformer"]

    sig = inspect.signature(ModelClass.__init__)
    init_kwargs = {
        k: v for k, v in params.items() if k in sig.parameters and k != "self"
    }

    model = ModelClass(**init_kwargs).to(device)
    logging.info("Model initialized.")
    log_model_architecture_and_params(model)

    # 8) criterion, optimizer, early stopping
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        eps=float(config["training"]["epsilon"])
    )
    early_stopper = EarlyStopping(
        patience=config["training"]["patience"],
        verbose=True,
        mode="min"
    )
    scaler = torch.cuda.amp.GradScaler()

    # 9) training loop
    best_accuracy = 0.0
    for epoch in range(config["training"]["epochs"]):
        logging.info(f"=== Epoch {epoch+1}/{config['training']['epochs']} ===")

        # -- Train --
        train_accuracy, train_loss = train_epoch(
            model, train_loader, optimizer, device, loss_fn, scaler
        )
        logging.info(f"[Train] Loss: {train_loss:.4f}  Accuracy: {train_accuracy:.4f}")

        # -- Validation (only loss & accuracy) --
        val_accuracy, val_loss, _, _ = eval_model(
            model,
            valid_loader,
            device,
            loss_fn,
            config["model"]["num_labels"]
        )
        logging.info(f"[Valid] Loss: {val_loss:.4f}  Accuracy: {val_accuracy:.4f}")

        if config.get("save_confusion_matrix", True):
            # still save confusion matrix if desired
            _, _, cm, _ = eval_model(
                model,
                valid_loader,
                device,
                loss_fn,
                config["model"]["num_labels"]
            )
            cm_path = os.path.join(out_dir, f"cm_epoch_{epoch+1}.png")
            plot_confusion_matrix(cm, classes=list(label2id.keys()), output_path=cm_path)
            logging.info(f"Saved confusion matrix to {cm_path}")

        # save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), config["paths"]["best_model_path"])
            logging.info(f"** New best model saved (Validation Accuracy: {best_accuracy:.4f}) **")

        # early stopping on val_loss
        early_stopper(val_loss, model, config["paths"]["best_model_path"])
        if early_stopper.early_stop:
            logging.info("Early stopping triggered.")
            break

    # 10) final test evaluation (compute all metrics)
    model.load_state_dict(torch.load(config["paths"]["best_model_path"]))
    model.eval()
    test_accuracy, test_loss, cm, test_metrics = eval_model(
        model,
        test_loader,
        device,
        loss_fn,
        config["model"]["num_labels"]
    )

    logging.info("=== Final Test Results ===")
    logging.info(f"Test Loss:      {test_loss:.4f}")
    logging.info(f"Test Accuracy:  {test_accuracy:.4f}")
    logging.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logging.info(f"Test Recall:    {test_metrics['recall']:.4f}")
    logging.info(f"Test F1 Score:  {test_metrics['f1']:.4f}")
    logging.info(f"Test AUC:       {test_metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
