# scripts/generate_submission.py

"""

This is an example usage for predicting n times (mean model).

"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from src.model import BaselineMLP
from src.config_loader import load_config
from src.submission_utils import get_control_cells, assemble_submission

def main():
    # config
    config = load_config("config/val_config.yaml")

    model_path = config["model_path"]
    esm2_path = config["esm2_path"]
    val_csv_path = config["val_csv_path"]
    train_h5ad = config["train_h5ad"]
    save_path = config["save_path"]
    n_control = config.get("n_control_cells", 1000)

    # load embeddings and valiadtion data
    print("Loading ESM2 embeddings and validation data")
    esm2_dict = torch.load(esm2_path, map_location="cpu")
    val_df = pd.read_csv(val_csv_path)
    val_df = val_df[val_df["target_gene"].isin(esm2_dict)]

    # load model
    print("Loading model")
    model = BaselineMLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # predict
    X_pred_list, obs_list = [], []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        gene = row["target_gene"]
        n = int(row["n_cells"])
        emb = esm2_dict[gene].unsqueeze(0).repeat(n, 1).float()
        with torch.no_grad():
            pred = model(emb).numpy().astype(np.float32)
        X_pred_list.append(pred)
        obs_list.extend([gene] * n)

    X_pred = np.vstack(X_pred_list)
    obs_df = pd.DataFrame({"target_gene": obs_list})

    # submission format
    print("Assembling submission")
    control_sample, var = get_control_cells(train_h5ad, n_control)
    adata_submit = assemble_submission(X_pred, obs_df, control_sample, var)

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    adata_submit.write_h5ad(save_path)
    print(f"Submission saved to: {save_path}")

if __name__ == "__main__":
    main()
