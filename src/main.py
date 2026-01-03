import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from src.utils import (
    auto_detect_splits,
    coco_to_metadata,
    get_hf_token,
    visualize_sample,
)

console = Console()


def main():
    """Main entry point for the COCO to HuggingFace format converter."""
    parser = argparse.ArgumentParser(
        description="Convert COCO format annotations to HuggingFace metadata JSONL files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing train/test/validation subdirectories",
    )
    parser.add_argument(
        "--train-annotations",
        type=Path,
        help="Optional: Path to training annotations JSON file (overrides auto-detection)",
    )
    parser.add_argument(
        "--test-annotations",
        type=Path,
        help="Optional: Path to test annotations JSON file (overrides auto-detection)",
    )
    parser.add_argument(
        "--validation-annotations",
        type=Path,
        help="Optional: Path to validation annotations JSON file (overrides auto-detection)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate sample visualization images with bounding boxes",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        metavar="REPO_ID",
        help="Push the dataset to HuggingFace Hub (format: username/dataset-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (defaults to HF_TOKEN env var or huggingface-cli login)",
    )
    args = parser.parse_args()

    # Print banner
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]COCO → HuggingFace Converter[/bold cyan]\n"
            "[dim]Convert COCO annotations to HF dataset format[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    if not args.data_dir.is_dir():
        console.print(
            f"[red]✗[/red] Error: {args.data_dir} is not a valid directory",
            style="bold red",
        )
        exit(1)

    # Auto-detect splits
    console.print("[cyan]📂 Detecting dataset splits...[/cyan]")
    detected_splits = auto_detect_splits(args.data_dir)

    # Override with manual paths if provided
    annotations = {
        "train": args.train_annotations or detected_splits["train"],
        "test": args.test_annotations or detected_splits["test"],
        "validation": args.validation_annotations or detected_splits["validation"],
    }

    # Process each split
    processed_count = 0
    for split_name, coco_path in annotations.items():
        if coco_path is None:
            continue

        if not coco_path.is_file():
            console.print(
                f"[yellow]⚠[/yellow]  Skipping {split_name}: annotations file not found at {coco_path}"
            )
            continue

        # Write metadata.jsonl in the split directory under data_dir
        console.print(f"\n[bold cyan]Processing {split_name} split[/bold cyan]")
        split_dir = args.data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_path = split_dir / "metadata.jsonl"
        categories = coco_to_metadata(coco_path, out_path)
        processed_count += 1
        console.print(f"[green]✓[/green] Wrote metadata to [cyan]{out_path}[/cyan]")

        # Optionally visualize a sample
        if args.visualize:
            console.print("[cyan]🎨 Generating visualization...[/cyan]")
            vis_output = Path("sample_visualization.jpg")
            visualize_sample(out_path, split_dir, vis_output, categories)

    if processed_count == 0:
        console.print()
        console.print(
            Panel(
                f"[yellow]No annotation files found or processed in {args.data_dir}[/yellow]\n\n"
                "[bold]Expected structure:[/bold]\n"
                "  data-dir/\n"
                "    ├── train/instances*.json\n"
                "    ├── validation/instances*.json\n"
                "    └── test/instances*.json (optional)\n\n"
                "[dim]Or use --train-annotations, --validation-annotations, --test-annotations to specify paths manually.[/dim]",
                title="[red]⚠ Warning[/red]",
                border_style="yellow",
            )
        )
        exit(1)

    # Load dataset and print info
    console.print()
    console.print("[bold cyan]📊 Loading and analyzing dataset...[/bold cyan]")

    # Get the first available annotation file to build dataset
    annotation_file = None
    for split_name in ["train", "validation", "test"]:
        ann_path = annotations[split_name]
        if ann_path and isinstance(ann_path, Path) and ann_path.is_file():
            annotation_file = ann_path
            break

    if not annotation_file:
        console.print(
            "[red]✗[/red] Error: No annotation file found to build dataset",
            style="bold red",
        )
        exit(1)

    # Load the dataset once
    from src.utils import load_dataset_helper

    # Type assertion for mypy/pyright
    assert isinstance(annotation_file, Path)
    try:
        dataset = load_dataset_helper(args.data_dir, annotation_file)

        # Show which splits were loaded
        console.print(f"[cyan]  Splits:[/cyan] {', '.join(dataset.keys())}")

        # Get class names
        first_split = list(dataset.keys())[0]
        class_names = dataset[first_split].features["objects"]["category"].feature.names
        console.print(f"[cyan]  Classes:[/cyan] {len(class_names)}")

        # Show first few classes as a preview
        preview = ", ".join(class_names[:8])
        if len(class_names) > 8:
            preview += f", ... ({len(class_names) - 8} more)"
        console.print(f"[dim]  Preview: {preview}[/dim]")

        # Show split sizes
        for split_name in dataset.keys():
            size = len(dataset[split_name])
            console.print(f"[dim]  {split_name}: {size:,} images[/dim]")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load dataset: {e}", style="bold red")
        exit(1)

    # Push to hub if requested
    if args.push_to_hub:
        console.print()
        console.print(
            "[bold cyan]🚀 Preparing to push to HuggingFace Hub...[/bold cyan]"
        )

        # Get token with fallback priority
        token = get_hf_token(args.token)
        if not token:
            console.print()
            console.print(
                Panel(
                    "[red]No HuggingFace token found![/red]\n\n"
                    "[bold]Please provide a token via one of these methods:[/bold]\n"
                    "  1. [cyan]--token YOUR_TOKEN[/cyan]\n"
                    "  2. Set [cyan]HF_TOKEN[/cyan] environment variable\n"
                    "  3. Run [cyan]huggingface-cli login[/cyan]",
                    title="[red]🔑 Authentication Required[/red]",
                    border_style="red",
                )
            )
            exit(1)

        # Push the already-loaded dataset
        console.print(
            f"[cyan]⬆️  Pushing dataset to [bold]{args.push_to_hub}[/bold]...[/cyan]"
        )
        dataset.push_to_hub(args.push_to_hub, token=token, private=False)
        console.print("[green]✓ Dataset successfully pushed![/green]")
        console.print(
            f"[cyan]🔗 View at: https://huggingface.co/datasets/{args.push_to_hub}[/cyan]"
        )

    # Success summary
    console.print()
    console.print(
        Panel(
            f"[green]Successfully processed {processed_count} split(s)![/green]\n"
            f"[dim]Output directory: {args.data_dir}[/dim]",
            title="[green]✓ Complete[/green]",
            border_style="green",
        )
    )
    console.print()


if __name__ == "__main__":
    main()
