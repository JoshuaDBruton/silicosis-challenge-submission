#!/usr/bin/env bash
set -e

if ! command -v uv &> /dev/null; then
    echo "uv not found, installing via pip..."
    python3 -m pip install --upgrade pip
    python3 -m pip install uv
else
    echo "uv is already installed."
fi

if [ ! -d ".venv" ]; then
    echo "Creating .venv with uv..."
    uv venv .venv
else
    echo ".venv already exists."
fi

echo "Installing requirements with uv..."
.venv/bin/uv pip install -r requirements.txt

if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
fi

if ! .venv/bin/python -m pip show gdown &> /dev/null; then
    echo "Installing gdown in the venv..."
    .venv/bin/pip install gdown
fi

# GDRIVE LINKS
GDRIVE_LINK_1="https://drive.google.com/file/d/1yg7tVOzwq6GVQv5yn7gOhH_Ct5-vlPGA/view?usp=drive_link"
GDRIVE_LINK_2="https://drive.google.com/file/d/1bymQIcvY0XkMkIJJC_ibIhVNCu0wjocX/view?usp=drive_link"
GDRIVE_LINK_3="https://drive.google.com/file/d/1znP_wpGYHX6kJxZEY-UtbSRZh1Xk312D/view?usp=drive_link"

# FILENAMES
DEST_1="checkpoints/c-c4.pth"
DEST_2="checkpoints/c-r3-126-0.pth"
DEST_3="checkpoints/open_clip_pytorch_model.bin"

echo "Downloading $GDRIVE_LINK_1 to $DEST_1"
.venv/bin/gdown "$GDRIVE_LINK_1" -O "$DEST_1"

echo "Downloading $GDRIVE_LINK_2 to $DEST_2"
.venv/bin/gdown "$GDRIVE_LINK_2" -O "$DEST_2"

echo "Downloading $GDRIVE_LINK_3 to $DEST_3"
.venv/bin/gdown "$GDRIVE_LINK_3" -O "$DEST_3"

echo "Setup complete!"

