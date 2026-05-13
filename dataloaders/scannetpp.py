import os
import glob
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

class NPZFolderTest(Dataset):
    def __init__(self, root: str, features: Optional[str] = None):
        super().__init__()
        self.root = root
        self.features = features
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = np.load(path)
        
        keys = list(data.keys())
        if 'noisy' in keys: points = data['noisy']
        elif 'points' in keys: points = data['points']
        else: points = data[keys[0]]

        features = data['features'] if (self.features is not None and 'features' in keys) else None

        # normalize
        points = points[:, :3]
        center = np.mean(points, axis=0)
        points -= center
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 1e-6: points /= scale

        res = {
            "idx": index,
            "train_points": torch.from_numpy(points).float(),
            "train_points_center": center,
            "train_points_scale": scale,
        }

        if features is not None:
            res["features"] = torch.from_numpy(features).float()

        return res


class ScanNetPP_NPZ(Dataset):
    def __init__(
        self,
        root: str,
        mode: str = "training",
        additional_features: bool = False,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = root
        self.mode = mode
        self.additional_features = additional_features
        self.augment = augment if mode == "training" else False
        self.transform = transform

        logger.info(f"[{mode.upper()}] Scanning for data recursively in: {self.root}")

        if not os.path.exists(self.root):
            logger.error(f"Root path does not exist: {self.root}")
            self.scene_batches = []
        else:
            search_pattern = os.path.join(self.root, "**", "*.npz")
            all_paths = sorted(glob.glob(search_pattern, recursive=True))
            
            self.scene_batches = []
            for path in all_paths:
                scene_name = os.path.basename(os.path.dirname(path))
                self.scene_batches.append({
                    "scene": scene_name,
                    "npz": path
                })

        logger.info(f"[{mode.upper()}] Loaded {len(self.scene_batches)} batches")
        
        if len(self.scene_batches) == 0:
             pass

    def __len__(self):
        return len(self.scene_batches)


class ScanNetPP(ScanNetPP_NPZ):
    def __init__(
        self,
        root: str,
        mode: str = "training",
        additional_features: bool = False,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=root, mode=mode, additional_features=additional_features, augment=augment, transform=transform
        )

    def __getitem__(self, index):
        batch_data = {}
        
        max_retries = 3
        for _ in range(max_retries):
            try:
                data_info = self.scene_batches[index]
                data_dict = np.load(data_info["npz"])
                
                keys = list(data_dict.keys())
                
                if 'clean' in keys: clean = data_dict["clean"]
                elif 'gt' in keys: clean = data_dict["gt"]
                else: clean = data_dict[keys[0]] 

                if 'noisy' in keys: noisy = data_dict["noisy"]
                elif 'points' in keys: noisy = data_dict["points"]
                else: noisy = clean 

                break
            except Exception as e:
                logger.warning(f"Failed to load {data_info['npz']}: {e}")
                index = np.random.randint(0, self.__len__())
        else:
            raise RuntimeError("Failed to load data after retries")

        # extract points (XYZ)
        points_noisy = noisy[:, :3]
        points_clean = clean[:, :3]

        if noisy.shape[1] > 3:
            batch_data["noisy_colors"] = torch.from_numpy(noisy[:, 3:]).float()
        if clean.shape[1] > 3:
            batch_data["clean_colors"] = torch.from_numpy(clean[:, 3:]).float()

        # append features
        if self.additional_features:
            if "features" in data_dict:
                features = data_dict["features"]
                batch_data["noisy_features"] = torch.from_numpy(features).float()
            else:
                # logger.warning(f"Feature missing in {data_info['npz']}")
                batch_data["noisy_features"] = torch.zeros((points_noisy.shape[0], 0)).float()

        # normalize
        if "center" not in data_dict:
            center = np.mean(points_noisy, axis=0)
            points_noisy -= center
            points_clean -= center
        else:
            center = data_dict["center"]

        if "scale" not in data_dict:
            scale = np.max(np.linalg.norm(points_noisy, axis=1))
            if scale < 1e-6: scale = 1.0
            points_noisy /= scale
            points_clean /= scale
        else:
            scale = data_dict["scale"]

        # augmentation
        if self.augment and np.random.rand() < 0.5:
            points_noisy, R = self.random_rotate(points_noisy)
            points_clean = points_clean @ R.T # Apply same rotation

        # shuffle
        rand_idxs = np.arange(points_noisy.shape[0])
        np.random.shuffle(rand_idxs)

        points_noisy = points_noisy[rand_idxs]
        points_clean = points_clean[rand_idxs]
        
        if "noisy_colors" in batch_data:
            batch_data["noisy_colors"] = batch_data["noisy_colors"][rand_idxs]
        if "clean_colors" in batch_data:
            batch_data["clean_colors"] = batch_data["clean_colors"][rand_idxs]
        if "noisy_features" in batch_data and batch_data["noisy_features"].shape[1] > 0:
            batch_data["noisy_features"] = batch_data["noisy_features"][rand_idxs]

        if self.transform is not None:
            points_noisy = self.transform(points_noisy)
            points_clean = self.transform(points_clean)

        batch_data["idx"] = index

        batch_data["noisy_points"] = torch.from_numpy(points_noisy).float()
        batch_data["clean_points"] = torch.from_numpy(points_clean).float()
        
        batch_data["center"] = center
        batch_data["scale"] = scale

        return batch_data

    def random_rotate(self, points):
        theta = np.random.uniform(0, 2 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        R = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
        return points @ R.T, R