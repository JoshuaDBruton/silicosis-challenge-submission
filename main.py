import yaml
import json
from dataset.dataset import DicomDataset
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from models import create_combined_model
import torch
import os


def main():
    # Load Silicosis Model
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

    # Load Dataset
    dataset = DicomDataset(imgpath="/media/joshua/Data/Silicosis Dataset/anonymised_silicosis_training_images-sep-25")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    label_descriptions = config["LABEL_DESCRIPTIONS"]["LABEL_SET_2"]

    # Inference Silicosis
    for image in dataset:
        print(image.shape)
        logits = model.forward(image, label_descriptions)
        print(logits)


if __name__ == "__main__":
    main()
