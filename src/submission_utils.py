# src/submission_utils.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import anndata

def get_control_cells(h5ad_path, n=1000):
    """
    Extract n non-targeting control cells from the training h5ad file.

    Returns:
        control_sample (AnnData): Subset AnnData of control cells
    """
    adata = anndata.read_h5ad(h5ad_path)
    control_cells = adata[adata.obs["target_gene"] == "non-targeting"]
    return control_cells[:n].copy(), adata.var.copy()

def assemble_submission(X_pred, obs_df, control_sample, var):
    """
    Combine prediction + control cells into a submission-ready AnnData object.

    Parameters:
        X_pred (ndarray): Predicted expression matrix
        obs_df (DataFrame): obs for predicted cells
        control_sample (AnnData): control AnnData
        var (DataFrame): var metadata from training data

    Returns:
        AnnData: combined submission object
    """
    X_control = (
        control_sample.X.toarray()
        if isinstance(control_sample.X, csr_matrix)
        else control_sample.X
    )
    X_final = np.vstack([X_pred, X_control.astype(np.float32)])
    obs_final = pd.concat([
        obs_df,
        pd.DataFrame({"target_gene": ["non-targeting"] * len(control_sample)})
    ], ignore_index=True)

    adata_submit = anndata.AnnData(X=X_final, obs=obs_final, var=var)
    return adata_submit
