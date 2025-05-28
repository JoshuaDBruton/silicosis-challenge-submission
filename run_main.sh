#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <imgpath>"
    exit 1
fi

source .venv/bin/activate

python main.py --imgpath "$1"
