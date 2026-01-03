import json
import random
import re
from collections import defaultdict
from pathlib import Path

import datasets
from datasets import ClassLabel, Features, Sequence, Value
from PIL import Image, ImageDraw
from tqdm import tqdm


def load_coco_data(coco_path: Path):
    """Load COCO annotation file and return the data."""
    with coco_path.open() as f:
        return json.load(f)


def extract_categories(data: dict) -> dict:
    """Extract category mapping from COCO data (contiguous index -> name).

    COCO category IDs have gaps (e.g., no 66-72), so we map them to
    contiguous 0-indexed IDs matching the order they appear.
    """
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    return {idx: cat["name"] for idx, cat in enumerate(categories)}


def find_instances_file(directory: Path) -> Path | None:
    """Find an instances*.json file in the given directory."""
    instances_pattern = re.compile(r"^instances.*\.json$")
    instances_files = [
        f
        for f in directory.iterdir()
        if f.is_file() and instances_pattern.match(f.name)
    ]
    return instances_files[0] if instances_files else None


def auto_detect_splits(data_dir: Path) -> dict[str, Path | None]:
    """Auto-detect train/test/validation directories and their annotations."""
    splits = {}

    for split_name in ["train", "test", "validation"]:
        split_dir = data_dir / split_name
        if split_dir.is_dir():
            instances_file = find_instances_file(split_dir)
            splits[split_name] = instances_file
        else:
            splits[split_name] = None

    return splits


def coco_to_metadata(coco_path: Path, out_path: Path):
    data = load_coco_data(coco_path)

    # Create mapping from COCO category IDs to contiguous 0-indexed IDs
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    coco_id_to_index = {cat["id"]: idx for idx, cat in enumerate(categories)}

    objects = defaultdict(lambda: {"bbox": [], "category": []})
    print(f"Processing annotations from {coco_path.name}...")
    for ann in tqdm(data.get("annotations", []), desc="Loading annotations"):
        img_id = ann["image_id"]
        objects[img_id]["bbox"].append([float(x) for x in ann["bbox"]])
        # Map COCO category ID to contiguous index
        contiguous_id = coco_id_to_index[ann["category_id"]]
        objects[img_id]["category"].append(contiguous_id)

    images = data.get("images", [])
    with out_path.open("w") as f:
        for img in tqdm(images, desc="Writing metadata"):
            img_id = img["id"]
            row = {
                "file_name": img["file_name"],
                "objects": objects.get(img_id, {"bbox": [], "category": []}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(images)} images)")

    return extract_categories(data)


def visualize_sample(
    metadata_path: Path, image_dir: Path, output_path: Path, categories: dict
):
    """Draw bounding boxes on a random sample image and save visualization."""
    # Read all metadata entries
    with metadata_path.open() as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if not entries:
        print(f"No entries found in {metadata_path}")
        return

    # Pick a random entry with objects
    entries_with_objects = [e for e in entries if e["objects"]["bbox"]]
    if not entries_with_objects:
        print(f"No images with annotations found in {metadata_path}")
        return

    sample = random.choice(entries_with_objects)
    image_path = image_dir / sample["file_name"]

    if not image_path.exists():
        print(f"Warning: Image not found at {image_path}")
        return

    # Load image and draw boxes
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Draw each bounding box
    for bbox, cat_id in zip(sample["objects"]["bbox"], sample["objects"]["category"]):
        x, y, w, h = bbox
        # COCO format is [x, y, width, height], convert to corner coordinates
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        # Draw category label if available
        if categories and cat_id in categories:
            cat_name = categories[cat_id]
            draw.text((x, y - 15), cat_name, fill="red")

    # Save visualization
    img.save(output_path)
    print(f"Saved visualization to {output_path}")


def build_features_from_coco(coco_path: Path) -> tuple[Features, list[str]]:
    data = json.loads(coco_path.read_text())
    # COCO categories: list of {"id": int, "name": str, ...}
    cats = sorted(data["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]

    features = Features(
        {
            "image": datasets.Image(),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"), length=4)),
                # use "category" or "categories" to match your metadata.jsonl key
                "category": Sequence(ClassLabel(names=names)),
            },
        }
    )
    return features, names


def load_dataset_helper(data_dir: Path, annotation_file: Path):
    """Load a HuggingFace dataset from COCO-format metadata.

    Args:
        data_dir: Directory containing the split subdirectories (train/val/test)
        annotation_file: Path to the COCO annotations file (to extract features)

    Returns:
        A HuggingFace DatasetDict with properly typed features
    """
    features, names = build_features_from_coco(annotation_file)
    dataset = datasets.load_dataset(
        "imagefolder",
        data_dir=str(data_dir),
        features=features,
    )
    return dataset


def get_hf_token(token_arg: str | None) -> str | None:
    """Get HuggingFace token from args, environment, or huggingface-cli.

    Args:
        token_arg: Token passed via CLI argument (takes priority)

    Returns:
        Token string or None if not found
    """
    import os

    # Priority 1: CLI argument
    if token_arg:
        return token_arg

    # Priority 2: Environment variable
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]

    # Priority 3: Try to get from huggingface_hub (if user ran `huggingface-cli login`)
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    return None


def push_to_hub_helper(
    data_dir: Path, annotation_file: Path, repo_id: str, token: str | None
):
    """Push dataset to HuggingFace Hub.

    Args:
        data_dir: Directory containing the dataset
        annotation_file: Path to COCO annotations for features
        repo_id: HuggingFace Hub repo ID (username/dataset-name)
        token: HuggingFace API token (optional if logged in)
    """
    # Load the dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = load_dataset_helper(data_dir, annotation_file)

    # Push to hub
    print(f"Pushing dataset to {repo_id}...")
    dataset.push_to_hub(repo_id, token=token, private=False)
    print(f"✓ Dataset successfully pushed to https://huggingface.co/datasets/{repo_id}")