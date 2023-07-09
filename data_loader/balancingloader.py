from pathlib import Path
import math

import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import rotate

import torch
from .augmentation import *


class CVbalancedTrainloader(object):
    def __init__(self, img_dir: str, cv_num: int = 0, seed: int = 0, num_shot: int = 0) -> None:
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
        self.gts = []
        for data_dir, gt_pos in zip(self.data_dirs, self.gt_poses):
            img_paths = sorted(data_dir.joinpath(f"imgs").glob("*.*"))
            gt_paths = sorted(data_dir.joinpath(f"hms").glob("*.png"))
            pos_frame = gt_pos[:, 2]
            self.img_paths.extend(
                list(zip([img_paths[int(idx) - 1] for idx in pos_frame], [img_paths[int(idx)] for idx in pos_frame]))
            )
            self.gt_paths.extend([gt_paths[int(idx) - 1] for idx in pos_frame])
            self.gts.extend(gt_pos)

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

    def random_crop_param(self, shape, pos):
        h, w = shape
        left = pos[0] - np.random.randint(0, self.crop_size[0])
        left = int(min(max(0, left), w - self.crop_size[0]))
        top = pos[1] - np.random.randint(0, self.crop_size[1])
        top = int(min(max(0, top), h - self.crop_size[1]))
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
            point = self.gts[data_id]

            top, bottom, left, right = self.random_crop_param(img.shape[:2], point[:2])
            img = img[top:bottom, left:right]
            img2 = img2[top:bottom, left:right]

            gt = gt[int(top / 4) : int(bottom / 4), int(left / 4) : int(right / 4)]

        img = torch.from_numpy(img.astype(np.float32))
        img2 = torch.from_numpy(img2.astype(np.float32))
        gt = torch.from_numpy(gt.astype(np.float32))
        return torch.stack([img, img2], axis=0), gt.unsqueeze(0)
