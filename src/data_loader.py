# src/data_loader.py

import torch
import pandas as pd
import anndata
import numpy as np
from torch.utils.data import Dataset

# Will change to cell-load.

def load_mean_expression(h5ad_path):
    """
    Load non-control cells and compute mean gene expression per target gene.

    Parameters:
        h5ad_path (str): Path to the training .h5ad file

    Returns:
        DataFrame: mean expression matrix (genes x target_gene)

    Note:
        For example usage. TO BE DELETED.
    """
    adata = anndata.read_h5ad(h5ad_path)
    adata = adata[adata.obs["target_gene"] != "non-targeting"]
    matrix = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    df = pd.DataFrame(matrix)
    df["target_gene"] = adata.obs["target_gene"].values
    mean_expr = df.groupby("target_gene").mean()
    return mean_expr

def build_training_dataset(mean_expr, esm2_dict):
    """
    Construct tensors from ESM2 embeddings and mean expression matrix.

    Parameters:
        mean_expr (pd.DataFrame): mean expression per target gene
        esm2_dict (dict): target_gene → ESM2 embedding (Tensor)

    Returns:
        Tuple[Tensor, Tensor]: (X_tensor, Y_tensor)
    """
    usable_genes = [g for g in mean_expr.index if g in esm2_dict]
    X_tensor = torch.stack([esm2_dict[g].float() for g in usable_genes])
    Y_tensor = torch.tensor(mean_expr.loc[usable_genes].values, dtype=torch.float32)
    return X_tensor, Y_tensor

class ESM2ExpressionDataset(Dataset):
    """
    A PyTorch Dataset wrapping ESM2 embeddings and corresponding gene expression.

    Each sample is a tuple: (ESM2_embedding, expression_vector)
    """
    def __init__(self, X_tensor, Y_tensor):
        self.X = X_tensor
        self.Y = Y_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
