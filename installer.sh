#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

if ! pip show gdown &> /dev/null; then
    pip install gdown
fi

# GDRIVE FILE IDS
GDRIVE_ID_1="1yg7tVOzwq6GVQv5yn7gOhH_Ct5-vlPGA"
GDRIVE_ID_2="1bymQIcvY0XkMkIJJC_ibIhVNCu0wjocX"
GDRIVE_ID_3="1znP_wpGYHX6kJxZEY-UtbSRZh1Xk312D"

# FILENAMES
DEST_1="checkpoints/c-c4.pth"
DEST_2="checkpoints/c-r3-126-0.pth"
DEST_3="checkpoints/open_clip_pytorch_model.bin"

echo "Downloading $GDRIVE_ID_1 to $DEST_1"
gdown "$GDRIVE_ID_1" -O "$DEST_1"

echo "Downloading $GDRIVE_ID_2 to $DEST_2"
gdown "$GDRIVE_ID_2" -O "$DEST_2"

echo "Downloading $GDRIVE_ID_3 to $DEST_3"
gdown "$GDRIVE_ID_3" -O "$DEST_3"

echo "Setup complete!"

