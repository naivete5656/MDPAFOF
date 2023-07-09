from pathlib import Path
import math

import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import rotate

import torch

from .augmentation import *


class CVTrainloader(object):
    def __init__(
        self,
        img_dir: str,
        cv_num: int = 0,
        seed: int = 0,
        num_shot: int = 0,
        crop_size: tuple = (256, 256),
    ) -> None:
        self.load_data_dirs(img_dir)

        # pop test_dir
        self.data_dirs.pop(cv_num)
        self.gt_poses.pop(cv_num)

        # pop val_dir
        if cv_num == 3:
            self.data_dirs.pop(0)
            self.gt_poses.pop(0)
        else:
            self.data_dirs.pop(cv_num)
            self.gt_poses.pop(cv_num)

        self.img_paths = []
        self.gt_paths = []
        for data_dir in self.data_dirs:
            img_paths = sorted(data_dir.joinpath(f"imgs").glob("*.*"))
            gt_paths = sorted(data_dir.joinpath(f"hms").glob("*.png"))
            self.img_paths.extend(list(zip(img_paths[:-1], img_paths[1:])))
            self.gt_paths.extend(gt_paths)

        self.crop_size = (512, 512)

    def load_data_dirs(self, img_dir):
        self.data_dirs = []
        self.gt_poses = []
        if Path(img_dir + "_test").exists():
            for cand_dir in Path(img_dir + "_test").iterdir():
                if cand_dir.is_dir():
                    self.data_dirs.append(cand_dir)
                    pos = np.loadtxt(f"{cand_dir}/{cand_dir.stem}_mit.txt", comments="%", encoding="shift-jis")
                    self.gt_poses.append(pos)
        for cand_dir in Path(img_dir).iterdir():
            if cand_dir.is_dir():
                self.data_dirs.append(cand_dir)
                pos = np.loadtxt(f"{cand_dir}/{cand_dir.stem}_mit.txt", comments="%", encoding="shift-jis")
                self.gt_poses.append(pos)

    def __len__(self) -> int:
        return len(self.img_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def __getitem__(self, data_id: int) -> tuple:
        img_path = self.img_paths[data_id][0]
        img = np.array(Image.open(img_path).convert("L"))
        img = img / 255

        img_path2 = self.img_paths[data_id][1]
        img2 = np.array(Image.open(img_path2).convert("L"))
        img2 = img2 / 255

        gt_path = self.gt_paths[data_id]
        gt = np.array(Image.open(gt_path).convert("L"))
        gt = gt / 255

        if self.crop_size is not None:
            top, bottom, left, right = self.random_crop_param(img.shape)
            img = img[top:bottom, left:right]
            img2 = img2[top:bottom, left:right]
            gt = gt[round(top / 4) : round(bottom / 4), round(left / 4) : round(right / 4)]

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        return torch.stack([img, img2], axis=0), gt.unsqueeze(0)


class CVValloader(CVTrainloader):
    def __init__(self, img_dir: str, cv_num: int = 0) -> None:
        self.load_data_dirs(img_dir)

        # pop test_dir
        self.data_dirs.pop(cv_num)
        self.gt_poses.pop(cv_num)
        # pop val_dir
        if cv_num == 3:
            self.data_dirs = [self.data_dirs.pop(0)]
            self.gt_poses = self.gt_poses.pop(0)
        else:
            self.data_dirs = [self.data_dirs.pop(cv_num)]
            self.gt_poses = self.gt_poses.pop(cv_num)

        self.img_paths = []
        self.gt_paths = []
        for data_dir in self.data_dirs:
            img_paths = sorted(data_dir.joinpath(f"imgs").glob("*.*"))
            gt_paths = sorted(data_dir.joinpath(f"hms").glob("*.png"))
            self.img_paths.extend(list(zip(img_paths[:-1], img_paths[1:])))
            self.gt_paths.extend(gt_paths)
        if ("Fluo-N2DL-HeLa" in Path(img_dir).stem) or ("Fluo-N2DH-SIM+" in Path(img_dir).stem):
            self.pad = True
        else:
            self.pad = False

    def __getitem__(self, data_id: int) -> tuple:
        img_path = self.img_paths[data_id][0]
        img = np.array(Image.open(img_path).convert("L"))
        img = img / 255

        img_path2 = self.img_paths[data_id][1]
        img2 = np.array(Image.open(img_path2).convert("L"))
        img2 = img2 / 255

        gt_path = self.gt_paths[data_id]
        gt = np.array(Image.open(gt_path).convert("L"))
        gt = gt / 255

        if self.pad:
            # data augumentation
            scale = math.ceil(img.shape[0] / 64), math.ceil(img.shape[1] / 64)
            pad_size = (0, scale[0] * 64 - img.shape[0]), (0, scale[1] * 64 - img.shape[1])
            img = np.pad(img, pad_size)
            img2 = np.pad(img2, pad_size)

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        return torch.stack([img, img2], axis=0), gt.unsqueeze(0)


class CVTestloader(CVTrainloader):
    def __init__(self, img_dir: str, cv_num: int = 0) -> None:
        self.load_data_dirs(img_dir)

        # pop test_dir
        self.data_dirs = [self.data_dirs.pop(cv_num)]
        self.gt_poses = self.gt_poses.pop(cv_num)

        self.img_paths = []
        self.gt_paths = []
        for data_dir in self.data_dirs:
            img_paths = sorted(data_dir.joinpath(f"imgs").glob("*.*"))
            gt_paths = sorted(data_dir.joinpath(f"hms").glob("*.png"))
            self.img_paths.extend(list(zip(img_paths[:-1], img_paths[1:])))
            self.gt_paths.extend(gt_paths)

        if ("Fluo-N2DL-HeLa" in Path(img_dir).stem) or ("Fluo-N2DH-SIM+" in Path(img_dir).stem):
            self.pad = True
        else:
            self.pad = False

    def __getitem__(self, data_id: int) -> tuple:
        img_path = self.img_paths[data_id][0]
        img = np.array(Image.open(img_path).convert("L"))
        img = img / 255

        img_path2 = self.img_paths[data_id][1]
        img2 = np.array(Image.open(img_path2).convert("L"))
        img2 = img2 / 255

        gt_path = self.gt_paths[data_id]
        gt = np.array(Image.open(gt_path).convert("L"))
        gt = gt / 255

        if self.pad:
            # data augumentation
            scale = math.ceil(img.shape[0] / 64), math.ceil(img.shape[1] / 64)
            pad_size = (0, scale[0] * 64 - img.shape[0]), (0, scale[1] * 64 - img.shape[1])
            img = np.pad(img, pad_size)
            img2 = np.pad(img2, pad_size)

            pad_size = (0, int(img.shape[0] / 4) - gt.shape[0]), (0, int(img.shape[1] / 4) - gt.shape[1])
        else:
            pad_size = None

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        return torch.stack([img, img2], axis=0), gt.unsqueeze(0), data_id


if __name__ == "__main__":
    train_dataset = CVTestloader("/mnt/d/main/Mitosis_detection/datas/ctc_preprocessed/Fluo-N2DL-HeLa", 0, 0, 5)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    for data in train_loader:
        print(1)

    # data_dir = "/mnt/d/main/Mitosis_detection/datas/ctc_preprocessed/Fluo-N2DH-SIM+"

    # for cv_num in range(4):
    #     print(f'cv number {cv_num}')

    #     dataloader = CVTrainloader(data_dir, cv_num=cv_num)
    #     dataloader = CVValloader(data_dir, cv_num=cv_num)
    #     dataloader = CVTestloader(data_dir, cv_num=cv_num)

    data_dir = "/mnt/d/main/Mitosis_detection/datas/bcell_preprocessed/normal_type"

    for cv_num in range(4):
        print(f"cv number {cv_num}")

        # dataloader = CVTrainloader(data_dir, cv_num=cv_num)
        dataloader = CVValloader(data_dir, cv_num=cv_num)
        # dataloader = CVTestloader(data_dir, cv_num=cv_num)
        for data in dataloader:
            print(1)
