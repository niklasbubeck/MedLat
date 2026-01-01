import torch
from torch.utils.data import Dataset
import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset
import torchvision.transforms as transforms

two_d_datasets = [
    name for name, info in INFO.items()
    if "3d" not in name
]

three_d_datasets = [
    name for name, info in INFO.items()
    if "3d" in name
]

def load_dataset(root, name, split, transform=None, download=True, size=224):
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass(root=root, split=split, transform=transform, download=download, as_rgb=True, size=size)

class MergedMedMNIST(Dataset):
    def __init__(self, root, split="train", include_dataset_id=True, transform=None, download=True, size=224, ds_names=two_d_datasets):
        self.datasets = []
        self.index_map = []   # (dataset_id, index_inside_dataset)
        self.include_dataset_id = include_dataset_id
        self.split = split

        # Load datasets (lightweight: only metadata + mmap arrays)
        for dataset_id, name in enumerate(ds_names):
            print(f"Loading dataset: {name}")
            ds = load_dataset(root, name, split, transform, download, size)
            self.datasets.append(ds)

            # Build mapping: one entry per sample
            for i in range(len(ds)):
                self.index_map.append((dataset_id, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_id, inner_idx = self.index_map[idx]
        ds = self.datasets[dataset_id]

        img, label = ds[inner_idx]  # MedMNIST handles lazy loading

        # chestmnist is multiclass we just neglect it 
        if ds.flag == "chestmnist":
            label = np.array([0])

        file_name = f"{ds.flag}_{self.split}_{inner_idx}"

        if self.include_dataset_id:
            return {"image": img, "label": label, "dataset_id": dataset_id, "sbj": idx, "file_name": file_name}
        else:
            return {"image": img, "label": label, "sbj": idx, "file_name": file_name}