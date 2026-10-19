#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$REPO_ROOT/data"
ARCHIVE_DIR="$DATA_ROOT/.downloads"

for command in wget unzip; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Required command not found: $command" >&2
        exit 1
    fi
done

mkdir -p "$ARCHIVE_DIR" "$DATA_ROOT/MMEB-train/images" "$DATA_ROOT/MMEB-eval"

download_and_extract() {
    local url="$1"
    local archive="$ARCHIVE_DIR/$2"
    local destination="$3"

    echo "Downloading $2"
    wget --continue --tries=5 --timeout=60 -O "$archive" "$url"
    unzip -oq "$archive" -d "$destination"
    rm -- "$archive"
}

TRAIN_SUBSETS=(
    A-OKVQA
    ChartQA
    DocVQA
    HatefulMemes
    ImageNet_1K
    InfographicsVQA
    N24News
    OK-VQA
    SUN397
    VOC2007
)

for subset in "${TRAIN_SUBSETS[@]}"; do
    download_and_extract \
        "https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/$subset.zip" \
        "train-$subset.zip" \
        "$DATA_ROOT/MMEB-train/images"
done

download_and_extract \
    "https://huggingface.co/datasets/TIGER-Lab/MMEB-eval/resolve/main/images.zip" \
    "eval-images.zip" \
    "$DATA_ROOT/MMEB-eval"

echo "MMEB train and eval images are ready in $DATA_ROOT"
