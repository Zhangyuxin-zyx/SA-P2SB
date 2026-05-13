import typing
from typing import Optional, Tuple
import os
from omegaconf import DictConfig
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .arkitscenes import ArkitNPZ
from .punet import get_dataset
from .scannetpp import ScanNetPP

try:
    from .pointcleannet import PointCleanNetDataset
except ImportError:
    pass

def save_iter(dataloader: DataLoader, sampler: Optional[DistributedSampler] = None) -> typing.Iterator:
    """Return a save iterator over the loader."""
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            if sampler is not None:
                sampler.set_epoch(sampler.epoch + 1)
            yield next(iterator)

def get_dataloader(
    opt: DictConfig, sampling: bool = False
) -> Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    """
    Return the training and testing dataloaders.
    """
    test_dataset = None
    collate_fn = None

    data_root = opt.data.get("root", opt.data.get("data_dir", None))
    if data_root is None:
        raise ValueError("Config file must specify 'data.root' or 'data.data_dir'")

    if opt.data.dataset == "ArKitPP":
        train_dataset = ArkitNPZ(
            root=data_root,
            mode="training",
            features=opt.data.point_features,
        )
        test_dataset = ArkitNPZ(
            root=data_root,
            mode="validation",
            features=opt.data.point_features,
        )

    elif opt.data.dataset == "ScanNetPP":
        train_dataset = ScanNetPP(
            root=data_root,
            mode="training",
            additional_features=opt.data.get("point_features", "none") != "none",
            augment=opt.data.get("augment", False),
        )
        test_dataset = ScanNetPP(
            root=data_root,
            mode="validation",
            additional_features=opt.data.get("point_features", "none") != "none",
            augment=opt.data.get("augment", False),
        )

    elif opt.data.dataset == "PUNet":
        train_dataset = get_dataset(dataset_root=data_root, split="train")
        test_dataset = get_dataset(dataset_root=data_root, split="test")

    elif opt.data.dataset == "PointCleanNet":
        try:
            train_dataset = PointCleanNetDataset(opt, split='train')
            test_dataset = PointCleanNetDataset(opt, split='validation')
        except NameError:
            raise NotImplementedError("PointCleanNet dataset file missing or import failed.")

    else:
        raise NotImplementedError(f"Dataset {opt.data.dataset} not implemented!")

    if opt.distribution_type == "multi":
        train_sampler = DistributedSampler(train_dataset, num_replicas=opt.global_size, rank=opt.local_rank) if train_dataset else None
        test_sampler = DistributedSampler(test_dataset, num_replicas=opt.global_size, rank=opt.local_rank) if test_dataset else None
    else:
        train_sampler = None
        test_sampler = None

    num_workers = int(opt.data.get("workers", opt.data.get("num_workers", 4)))

    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=opt.training.bs if not sampling else opt.sampling.bs,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        if train_dataset is not None
        else None
    )

    test_dataloader = (
        DataLoader(
            test_dataset,
            batch_size=opt.training.bs if not sampling else opt.sampling.bs,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            generator=Generator().manual_seed(opt.training.seed),
            collate_fn=collate_fn,
        )
        if test_dataset is not None
        else None
    )

    return train_dataloader, test_dataloader, train_sampler, test_sampler