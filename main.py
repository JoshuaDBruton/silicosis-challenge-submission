import yaml
import json
import numpy as np
from tqdm import tqdm
from dataset.dataset import DicomDataset
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from models import create_combined_model
import torch
import os


def main():
    model_name = "biomedclip_local"

    with open("checkpoints/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)

    model, preprocess = create_combined_model(tokenizer, model_name, pretrained_path="checkpoints/open_clip_pytorch_model.bin", image_size=1792, use_qformer=True, use_swin=True)
    model.load_state_dict(torch.load(os.path.join("checkpoints", "chosen-checkpoints4.pth")))
    model.eval()

    dataset = DicomDataset(imgpath="/media/joshua/Data/Silicosis Dataset/anonymised_silicosis_training_images-sep-25")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    silicosis_label_descriptions = config["LABEL_DESCRIPTIONS"]["LABEL_SET_2"]
    tuberculosis_label_descriptions = config["LABEL_DESCRIPTIONS"]["BINARY_TUBERCULOSIS"]

    silicosis_logits = []
    for image in tqdm(dataset):
        silicosis_logit = model.forward(image, silicosis_label_descriptions)
        silicosis_logit = silicosis_logit.detach().cpu().numpy()
        silicosis_logits.append(silicosis_logit)

    model.load_state_dict(torch.load(os.path.join("checkpoints", "chosen-results3-126-0.pth")))
    model.eval()

    tuberculosis_logits = []
    for image in tqdm(dataset):
        tuberculosis_logit = model.forward(image, tuberculosis_label_descriptions)
        tuberculosis_logit = tuberculosis_logit.detach().cpu().numpy()
        tuberculosis_logits.append(tuberculosis_logit)


if __name__ == "__main__":
    main()
