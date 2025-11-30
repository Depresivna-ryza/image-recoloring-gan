from pathlib import Path
import random
from torch import Tensor
from torch.utils.data import Dataset
import os
from typing import Tuple, List
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class RecoloringDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: Tuple[int, int] = (512, 1024), hflip: bool = False):
        self.gray_root = os.path.join(root_dir, "img_gray")
        self.rgb_root = os.path.join(root_dir, "img_rgb")
        self.image_size = image_size
        self.hflip = hflip
        
        self.gray_mean = (0.27684267461299894,)
        self.gray_std = (0.22140119224786758,)
        self.rgb_mean = (0.3296253949403763, 0.24600205868482589, 0.29721091985702514)
        self.rgb_std = (0.235480435192585, 0.2151222825050354, 0.23368016630411148)
        
        self.image_paths = self._scan_image_pairs()

        self.gray_transform = transforms.Compose([
            # transforms.Resize(self.image_size),
            transforms.ToTensor(),
            # transforms.Normalize(self.gray_mean, self.gray_std),
            # transforms.Normalize((0.5, ), (0.5, ))
    
        ])

        self.rgb_transform = transforms.Compose([
            # transforms.Resize(self.image_size),
            transforms.ToTensor(),
            # transforms.Normalize(self.rgb_mean, self.rgb_std),
        ])
        
        print(f"Foundd {len(self.image_paths)} image pairs.")

    def _scan_image_pairs(self) -> List[Tuple[str, str]]:
        image_pairs = []

        for dirpath, _, filenames in os.walk(self.gray_root):
            for dirpath, _, filenames in os.walk(self.rgb_root):
                if not dirpath.startswith(self.gray_root):
                    continue
            # print(f"Scanning directory: {dirpath}")
            # print(f"Found {len(filenames)} files")
            # print(f"Files: {filenames}")
            for fname in filenames:
                if not fname.endswith(".png") :
                    continue

                subpath = os.path.relpath(dirpath, self.gray_root)
                gray_path = os.path.join(self.gray_root, subpath, fname)
                rgb_path = os.path.join(self.rgb_root, subpath, fname)

                if os.path.exists(rgb_path):
                    image_pairs.append((gray_path, rgb_path))

        return image_pairs

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        gray_path, rgb_path = self.image_paths[idx]

        with Image.open(gray_path) as gray_img:
            gray_img = gray_img.convert("L")

        with Image.open(rgb_path) as rgb_img:
            rgb_img = rgb_img.convert("RGB")
            
        if self.hflip and random.random() > 0.5: 
            gray_img = transforms.functional.hflip(gray_img)
            rgb_img = transforms.functional.hflip(rgb_img)
            
        gray_tensor = self.gray_transform(gray_img)
        rgb_tensor = self.rgb_transform(rgb_img)

        return gray_tensor, rgb_tensor
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    data_root = "data_col_public/"

    dataset = RecoloringDataset(Path(data_root))
    
    print(len(dataset))  # Number of image pairs
    print(dataset[0])  # First image pair (gray, rgb)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    gray_batch, rgb_batch = next(iter(dataloader))
    print(f"Gray batch shape: {gray_batch.shape}")
    print(f"RGB batch shape : {rgb_batch.shape}") 

    def show_images(gray_tensor, rgb_tensor):
        unnormalize_gray = gray_tensor * 0.5 + 0.5
        unnormalize_rgb = rgb_tensor * 0.5 + 0.5

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(unnormalize_gray.squeeze().cpu(), cmap='gray')
        axs[0].set_title("Grayscale Input")
        axs[1].imshow(unnormalize_rgb.permute(1, 2, 0).cpu())
        axs[1].set_title("RGB Target")
        for ax in axs:
            ax.axis("off")
        plt.savefig("TMP_sample_images.png")
        

    show_images(gray_batch[0], rgb_batch[0])