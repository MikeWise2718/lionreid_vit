from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from rich.console import Console
from rich.table import Table

console = Console()

IMG_SIZE = 384


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


def get_train_transform() -> transforms.Compose:
    """Training transforms: augmentation then grayscale + normalize."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_eval_transform() -> transforms.Compose:
    """Validation/test transforms: resize, grayscale, normalize only."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def split_dataset(
    image_paths: list[str],
    labels: list[int],
    seed: int = 42,
) -> dict[str, tuple[list[str], list[int]]]:
    """Stratified 70/15/15 train/val/test split. Returns dict with 'train', 'val', 'test' keys."""
    paths_train, paths_temp, labels_train, labels_temp = train_test_split(
        image_paths, labels, test_size=0.30, random_state=seed, stratify=labels,
    )
    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_temp, labels_temp, test_size=0.50, random_state=seed, stratify=labels_temp,
    )
    return {
        "train": (paths_train, labels_train),
        "val": (paths_val, labels_val),
        "test": (paths_test, labels_test),
    }


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build train/val/test DataLoaders. Returns (train_dl, val_dl, test_dl, class_names)."""
    image_paths, labels, class_names = discover_dataset(data_dir)
    print_dataset_stats(image_paths, labels, class_names)

    splits = split_dataset(image_paths, labels, seed=seed)

    console.print(
        f"\n[bold]Split:[/] train={len(splits['train'][0])}  "
        f"val={len(splits['val'][0])}  test={len(splits['test'][0])}"
    )

    train_ds = LionDataset(*splits["train"], transform=get_train_transform())
    val_ds = LionDataset(*splits["val"], transform=get_eval_transform())
    test_ds = LionDataset(*splits["test"], transform=get_eval_transform())

    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl, class_names
