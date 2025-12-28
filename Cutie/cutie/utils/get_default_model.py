"""
A helper function to get a default model for quick testing
"""
import os
from pathlib import Path
from omegaconf import open_dict
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import torch
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.utils.download_models import download_models_if_needed


def get_default_model() -> CUTIE:
    # Resolve config dir relative to this file
    CONFIG_DIR = (Path(__file__).resolve().parent.parent / "config").as_posix()

    # Initialize Hydra only once per process
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(version_base="1.3.2", config_dir=CONFIG_DIR)

    # Safe to call multiple times
    cfg = compose(config_name="eval_config")

    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg["weights"] = os.path.join(weight_dir, "cutie-base-mega.pth")
    get_dataset_cfg(cfg)

    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights, map_location="cuda")
    cutie.load_weights(model_weights)

    return cutie
