# COCO to HuggingFace Format Converter

![Test & Lint](https://github.com/benjamintli/coco2hf/workflows/Test%20%26%20Lint/badge.svg)
![CLI Smoke Test](https://github.com/benjamintli/coco2hf/workflows/CLI%20Smoke%20Test/badge.svg)


Convert COCO format annotations to HuggingFace dataset metadata format (JSONL).

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Usage

After installation, you can use the `coco2hf` command:

```bash
# Auto-detect annotations in train/validation/test directories
coco2hf --data-dir /path/to/coco2017

# Manually specify annotation files
coco2hf --data-dir /path/to/coco2017 \
  --train-annotations /path/to/instances_train2017.json \
  --validation-annotations /path/to/instances_val2017.json

# Generate sample visualizations
coco2hf --data-dir /path/to/coco2017 --visualize

# Push to HuggingFace Hub
coco2hf --data-dir /path/to/coco2017 \
  --train-annotations /path/to/instances_train2017.json \
  --push-to-hub username/my-coco-dataset
```

Or run directly with Python (from the virtual environment):

```bash
source .venv/bin/activate
python -m src.main --data-dir /path/to/coco2017
```

## Expected Directory Structure

```
data-dir/
├── train/
│   ├── instances*.json  (auto-detected)
│   └── *.jpg            (images)
├── validation/
│   ├── instances*.json  (auto-detected)
│   └── *.jpg            (images)
└── test/               (optional)
    ├── instances*.json
    └── *.jpg
```

## Output

The tool generates `metadata.jsonl` files in each split directory:

```
data-dir/
├── train/
│   └── metadata.jsonl
└── validation/
    └── metadata.jsonl
```

Each line in `metadata.jsonl` contains:
```json
{
  "file_name": "image.jpg",
  "objects": {
    "bbox": [[x, y, width, height], ...],
    "category": [0, 1, ...]
  }
}
```

## Options

- `--data-dir`: Root directory containing train/validation/test subdirectories (required)
- `--train-annotations`: Path to training annotations JSON (optional)
- `--validation-annotations`: Path to validation annotations JSON (optional)
- `--test-annotations`: Path to test annotations JSON (optional)
- `--visualize`: Generate sample visualization images with bounding boxes
- `--push-to-hub`: Push dataset to HuggingFace Hub (format: `username/dataset-name`)
- `--token`: HuggingFace API token (optional, defaults to `HF_TOKEN` env var or `huggingface-cli login`)

### Authentication for Hub Push

When using `--push-to-hub`, the tool looks for your HuggingFace token in this order:

1. `--token YOUR_TOKEN` (CLI argument)
2. `HF_TOKEN` environment variable
3. Token from `huggingface-cli login`

If no token is found, you'll get a helpful error message with instructions.
