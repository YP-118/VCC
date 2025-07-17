# src/config_loader.py
import yaml

def load_config(path):
    """
    Load YAML config file into a Python dict.

    Parameters:
        path (str): Path to the .yaml file

    Returns:
        dict: Configuration dictionary
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
