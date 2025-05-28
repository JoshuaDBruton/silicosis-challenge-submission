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
import csv


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    model_name = "biomedclip_local"

    with open("checkpoints/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    if (
        not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None
    ):
        _MODEL_CONFIGS[model_name] = model_cfg

    tokenizer = get_tokenizer(model_name)

    model, preprocess = create_combined_model(
        tokenizer,
        model_name,
        pretrained_path="checkpoints/open_clip_pytorch_model.bin",
        image_size=1792,
        use_qformer=True,
        use_swin=True,
    )
    model.load_state_dict(
        torch.load(os.path.join("checkpoints", "c-c4.pth"))
    )
    model.eval()

    dataset = DicomDataset(
        imgpath="/media/joshua/Data/Silicosis Dataset/anonymised_silicosis_training_images-sep-25"
    )

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    silicosis_label_descriptions = config["LABEL_DESCRIPTIONS"]["LABEL_SET_2"]
    tuberculosis_label_descriptions = config["LABEL_DESCRIPTIONS"]["BINARY_TUBERCULOSIS"]

    silicosis_logits = []
    image_names = []
    for image, image_name in tqdm(dataset, desc="Silicosis"):
        with torch.no_grad():
            silicosis_logit = model.forward(image, silicosis_label_descriptions)
        silicosis_logit = silicosis_logit.detach().cpu().numpy().flatten()
        silicosis_logits.append(silicosis_logit)
        image_names.append(image_name)

    model.load_state_dict(
        torch.load(os.path.join("checkpoints", "c-r3-126-0.pth"))
    )
    model.eval()

    tuberculosis_logits = []
    for image, image_name in tqdm(dataset, desc="TB"):
        with torch.no_grad():
            tuberculosis_logit = model.forward(image, tuberculosis_label_descriptions)
        tuberculosis_logit = tuberculosis_logit.detach().cpu().numpy().flatten()
        tuberculosis_logits.append(tuberculosis_logit)

    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["image_name", "Silicosis confidence", "TB confidence"]
        )

        for idx, image_name in enumerate(image_names):
            silico_logit = silicosis_logits[idx]
            tb_logit = tuberculosis_logits[idx]

            silico_probs = softmax(silico_logit)
            silico_conf = silico_probs[-2:].sum()

            tb_probs = softmax(tb_logit)
            tb_conf = tb_probs[1]  # index 1 is "positive" class

            writer.writerow([image_name, silico_conf, tb_conf])

    print("CSV file 'results.csv' generated.")


if __name__ == "__main__":
    main()

