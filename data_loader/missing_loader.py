import numpy as np
from PIL import Image
from skimage.draw import disk
from skimage.filters import gaussian
from pathlib import Path

import torch
from augmentation import *


class MissingSuperLoader(object):
    def __init__(
        self,
        img_dir: str,
        cv_num: int = 0,
        seed: int = 0,
        missing_rate: str = "0.05",
    ) -> None:
        self.load_data_dirs(img_dir)

        self.crop_size = (256, 256)

        # pop test_dir
        self.data_dirs.pop(cv_num)

        # pop train_dir
        if cv_num == 3:
            self.data_dirs.pop(0)
        else:
            self.data_dirs.pop(cv_num)

        self.img_paths = []
        self.gtposes = []
        self.gt_paths = []

        for img_dir in self.data_dirs:
            if img_dir.is_dir():
                img_paths = sorted(img_dir.joinpath(f"imgs").glob("*.*"))
                gt_paths = sorted(img_dir.joinpath(f"missing_{missing_rate}_new/{seed:02d}/hms").glob("*.*"))
                self.gt_paths.extend(gt_paths)
                select_gt_pos = np.loadtxt(f"{img_dir}/missing_{missing_rate}_new/{seed:02d}.txt", ndmin=2)
                select_gt_pos = select_gt_pos.astype(np.int16)

                for gt_path in gt_paths:
                    frame = int(gt_path.stem[1:])
                    gt_pos = select_gt_pos[select_gt_pos[:, 2] == frame]
                    self.gtposes.append(gt_pos)
                    self.img_paths.append([img_paths[frame - 1], img_paths[frame]])
        self.img_shape = Image.open(img_paths[0]).convert("L").size

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


class MissingCP(object):
    def __init__(
        self,
        img_dir: str,
        cv_num: int = 0,
        seed: int = 0,
        missing_rate: str = "0.05",
        hw_range: int = 40,
        sigma: int = 3,
        peak=False,
    ) -> None:
        self.load_data_dirs(img_dir, cv_num)

        self.crop_size = (512, 512)
        self.aug = True
        self.hw_range = hw_range
        self.peaks = peak

        self.img_paths = []
        self.crop_mit = []
        for img_dir in self.data_dirs:
            if img_dir.is_dir():
                img_paths = sorted(img_dir.joinpath(f"imgs").glob("*.*"))
                self.img_paths.extend(zip(img_paths[1:], img_paths[:-1]))

                select_gt_pos = np.loadtxt(f"{img_dir}/missing_{missing_rate}_new/{seed:02d}.txt", ndmin=2)
                select_gt_pos = select_gt_pos.astype(np.int16)

                for x, y, t, _ in select_gt_pos:
                    img_path = img_paths[t - 1]
                    img = np.array(Image.open(img_path).convert("L"))
                    img = np.pad(img, hw_range, "symmetric")
                    crop_img = img[y : y + 2 * hw_range, x : x + 2 * hw_range] / 255
                    img_path = img_paths[t]
                    img = np.array(Image.open(img_path).convert("L"))
                    img = np.pad(img, hw_range, "symmetric")
                    crop_img1 = img[y : y + 2 * hw_range, x : x + 2 * hw_range] / 255

                    self.crop_mit.append(np.stack([crop_img, crop_img1]))

        self.temp_hm = self.make_hm(hw_range / 2, sigma)

        self.brending_mask = np.zeros((2 * hw_range, 2 * hw_range))
        rr, cc = disk((hw_range, hw_range), int(hw_range * 3 / 4), shape=self.brending_mask.shape)
        self.brending_mask[rr, cc] = 1

    def __getitem__(self, data_id: int) -> tuple:
        img_path = self.img_paths[data_id][0]
        img = np.array(Image.open(img_path).convert("L"))
        img = img / 255

        img_path = self.img_paths[data_id][1]
        img2 = np.array(Image.open(img_path).convert("L"))
        img2 = img2 / 255

        if self.crop_size is not None:
            top, bottom, left, right = self.random_crop_param(img.shape)
            img = img[top:bottom, left:right]
            img2 = img2[top:bottom, left:right]

        img_size = img.shape[0]
        img = np.pad(img, self.hw_range, "constant")
        img2 = np.pad(img2, self.hw_range, "constant")

        gt = np.zeros((int(img_size / 4) + int(self.hw_range / 2), int(img_size / 4) + int(self.hw_range / 2)))
        num_paste = np.random.randint(1, 10)
        select_ids = np.random.choice(len(self.crop_mit), num_paste)
        pasted_poses = []
        for select_id in select_ids:
            crop1, crop2 = self.crop_mit[select_id]
            crop1, crop2, _ = self.random_aug(crop1, crop2)
            w = np.random.randint(img_size - self.hw_range)
            h = np.random.randint(img_size - self.hw_range)
            sigma = np.random.randint(1, 20)
            brending_mask = gaussian(self.brending_mask, sigma=sigma, truncate=1)
            img[w : w + 2 * self.hw_range, h : h + 2 * self.hw_range] = (
                crop1 * brending_mask + (1 - brending_mask) * img[w : w + 2 * self.hw_range, h : h + 2 * self.hw_range]
            )
            img2[w : w + 2 * self.hw_range, h : h + 2 * self.hw_range] = (
                crop2 * brending_mask + (1 - brending_mask) * img2[w : w + 2 * self.hw_range, h : h + 2 * self.hw_range]
            )

            w, h = round(w / 4), round(h / 4)
            gt[w : w + int(self.hw_range / 2), h : h + int(self.hw_range / 2)] = np.maximum(
                self.temp_hm, gt[w : w + int(self.hw_range / 2), h : h + int(self.hw_range / 2)]
            )
            pasted_poses.append([w, h])

        img, img2, gt = self.random_aug(img, img2, gt)
        # return img, img2, gt

        img = torch.from_numpy(img.astype(np.float32))[self.hw_range : -self.hw_range, self.hw_range : -self.hw_range]
        img2 = torch.from_numpy(img2.astype(np.float32))[self.hw_range : -self.hw_range, self.hw_range : -self.hw_range]
        gt = torch.from_numpy(gt.astype(np.float32))[
            round(self.hw_range / 4) : -round(self.hw_range / 4), round(self.hw_range / 4) : -round(self.hw_range / 4)
        ]
        pasted_poses = torch.from_numpy(np.array(pasted_poses))

        if self.peaks:
            return torch.stack([img, img2], axis=0), gt.unsqueeze(0), pasted_poses
        else:
            return torch.stack([img, img2], axis=0), gt.unsqueeze(0), data_id

    def load_data_dirs(self, img_dir, cv_num):
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
        # pop test_dir
        self.data_dirs.pop(cv_num)

        # pop train_dir
        if cv_num == 3:
            self.data_dirs.pop(0)
        else:
            self.data_dirs.pop(cv_num)

    def make_hm(self, hw_range, sigma):
        x = np.arange(0, hw_range, 1, np.float32)
        shift_y, shift_x = np.meshgrid(x, x)
        x0 = y0 = (hw_range) // 2
        hm = np.exp(-((shift_x - x0) ** 2 + (shift_y - y0) ** 2) / (2 * sigma**2))
        return hm

    def __len__(self) -> int:
        return len(self.img_paths)

    def random_crop_param(self, shape):
        h, w = shape
        top = np.random.randint(0, h - self.crop_size[0])
        left = np.random.randint(0, w - self.crop_size[1])
        bottom = top + self.crop_size[0]
        right = left + self.crop_size[1]
        return top, bottom, left, right

    def random_aug(self, img, img2, gt=np.ones((5, 5))):
        img, img2, gt = random_flip(img, img2, gt)
        img, img2, gt = random_rot(img, img2, gt)
        if self.aug:
            img, img2 = random_gaussian(img, img2)
            img, img2 = random_brightness(img, img2)
            img, img2 = random_elastic_transform(img, img2)
        return img, img2, gt
