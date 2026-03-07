from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from rich.console import Console
from rich.table import Table

console = Console()


def discover_dataset(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    """Walk the dataset directory and return (image_paths, labels, class_names).

    Expects structure: data_dir/<LionID>/<sighting_session>/*.JPG
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    image_paths: list[str] = []
    labels: list[int] = []

    for label, class_dir in enumerate(class_dirs):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in exts:
                image_paths.append(str(img_path))
                labels.append(label)

    return image_paths, labels, class_names


def print_dataset_stats(image_paths: list[str], labels: list[int], class_names: list[str]):
    """Print a rich table summarizing the dataset."""
    table = Table(title="Dataset Summary")
    table.add_column("Label", justify="right")
    table.add_column("Lion ID", style="cyan")
    table.add_column("Images", justify="right", style="green")

    from collections import Counter
    counts = Counter(labels)
    for label, name in enumerate(class_names):
        table.add_row(str(label), name, str(counts[label]))

    table.add_section()
    table.add_row("", "[bold]Total[/]", f"[bold]{len(image_paths)}[/]")
    console.print(table)


class LionDataset(Dataset):
    """PyTorch dataset for lion re-identification."""

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
