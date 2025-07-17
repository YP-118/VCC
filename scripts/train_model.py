# scripts/train_model.py

"""

This is an example usage for predicting using mean.

"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import BaselineMLP
from src.data_loader import (
    load_mean_expression,
    build_training_dataset,
    ESM2ExpressionDataset
)
from src.config_loader import load_config

def main():
    # config
    config = load_config("config/train_config.yaml")

    train_data_path = config["train_data"]
    esm2_path = config["esm2_path"]
    save_path = config["save_path"]

    input_dim = config["model"]["input_dim"]
    output_dim = config["model"]["output_dim"]
    dropout_rate = config["model"]["dropout_rate"]

    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["epochs"]
    learning_rate = config["training"]["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = list(range(torch.cuda.device_count()))
    print(f"Using {len(device_ids)} GPUs: {device_ids}")

    print("Loading training expression data")
    mean_expr = load_mean_expression(train_data_path)

    print("Loading ESM2 features")
    esm2_dict = torch.load(esm2_path, map_location="cpu")

    X_tensor, Y_tensor = build_training_dataset(mean_expr, esm2_dict)
    train_dataset = ESM2ExpressionDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # === Initialize model ===
    model = BaselineMLP(input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout_rate)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    # === Train loop ===
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}]"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch} | Train Loss: {total_loss:.4f}")

    # === Save model ===
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.module.state_dict(), save_path)
    print(f"🎉 Model saved to {save_path}")

if __name__ == "__main__":
    main()
