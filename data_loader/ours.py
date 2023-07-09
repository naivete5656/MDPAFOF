from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import disk
from skimage.filters import gaussian

import torch

from .augmentation import *


class FrameOrderFlippingAlphaBrendingPastingLoader(object):
    def __init__(
        self,
        img_dir: str,
        cv_num: int = 0,
        seed: int = 0,
        num_shot: int = 5,
        hw_range: int = 40,
        sigma: int = 3,
        peak=False,
    ) -> None:
        self.crop_size = (512, 512)
        self.aug = True
        self.hw_range = hw_range
        self.peaks = peak

        self.data_dirs = []
        self.gt_poses = []
        self.load_data_dirs(img_dir, cv_num)

        # prepare_frame_order_flipped_path_and_crop_around_annotation
        self.img_paths = []
        self.crop_mit = []
        for img_dir in self.data_dirs:
            if img_dir.is_dir():
                img_paths = sorted(img_dir.joinpath(f"imgs").glob("*.*"))

                # extend frame order flipped image path. The first and second col. start at frame = 1 and frame = 0.
                self.img_paths.extend(zip(img_paths[1:], img_paths[:-1]))

                # crop around annotation
                select_gt_pos = np.loadtxt(f"{img_dir}/fs/fs_{num_shot}/{seed:02d}.txt", ndmin=2)
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

                    # append cropped mitosis image to crop mit list
                    self.crop_mit.append(np.stack([crop_img, crop_img1]))

        # generate heatmap template. It is used for gt generation
        self.temp_hm = self.make_hm(hw_range / 2, sigma)

        # generate brending mask which is used for alpha-brending-pasting
        self.brending_mask = np.zeros((2 * hw_range, 2 * hw_range))
        rr, cc = disk((hw_range, hw_range), int(hw_range * 3 / 4), shape=self.brending_mask.shape)
        self.brending_mask[rr, cc] = 1

    def load_data_dirs(self, img_dir, cv_num):
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

    def __getitem__(self, data_id: int) -> tuple:
        # load image at frame t
        img_path = self.img_paths[data_id][0]
        img = np.array(Image.open(img_path).convert("L"))
        img = img / 255

        # load image at frame t-1
        img_path = self.img_paths[data_id][1]
        img2 = np.array(Image.open(img_path).convert("L"))
        img2 = img2 / 255

        # random cropping
        if self.crop_size is not None:
            top, bottom, left, right = self.random_crop_param(img.shape)
            img = img[top:bottom, left:right]
            img2 = img2[top:bottom, left:right]

        img_size = img.shape[0]
        img = np.pad(img, self.hw_range, "constant")
        img2 = np.pad(img2, self.hw_range, "constant")

        # generate ground truth and do alpha-brending-pasting
        gt = np.zeros((int(img_size / 4) + int(self.hw_range / 2), int(img_size / 4) + int(self.hw_range / 2)))

        # set number of pasting mitosis image
        num_paste = np.random.randint(1, 10)
        select_ids = np.random.choice(len(self.crop_mit), num_paste)
        pasted_poses = []
        for select_id in select_ids:
            # load mitosis cell image
            crop1, crop2 = self.crop_mit[select_id]

            crop1, crop2, _ = self.random_aug(crop1, crop2)

            # set random position
            w = np.random.randint(img_size - self.hw_range)
            h = np.random.randint(img_size - self.hw_range)

            # generate brending mask with gaussian blur with random sigme value
            sigma = np.random.randint(1, 20)
            brending_mask = gaussian(self.brending_mask, sigma=sigma, truncate=1)

            # carry out alpha-brending-pasting
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


class SupervisedOurloader(FrameOrderFlippingAlphaBrendingPastingLoader):
    def __init__(
        self,
        img_dir: str,
        cv_num: int = 0,
        seed: int = 0,
        num_shot: int = 5,
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

                select_gt_pos = np.loadtxt(f"{img_dir}/{img_dir.stem}_mit.txt", comments="%", encoding="shift-jis")
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

        return torch.stack([img, img2], axis=0), gt.unsqueeze(0)


if __name__ == "__main__":
    # for data_path in ['ctc_preprocessed/Fluo-N2DH-SIM+', 'ctc_preprocessed/Fluo-N2DL-HeLa', 'bcell_preprocessed/normal_type']:
    for data_path in ["ctc_preprocessed/Fluo-N2DL-HeLa"]:
        data_dir = f"./datas/{data_path}"

        cell_type = data_path.split("/")[1]

        data_loader = CVCPLoaderBrend2(data_dir)
        loader_name = "CVCPLoaderBrend2"
        # for loader_name in ['CVCPLoaderBrend_ver2', 'CVCPLoaderdifaug', 'CVCPLoaderflip']:
        # data_loader = eval(loader_name)(data_dir)

        save_path = Path(f"/home/kazuya/hdd/Mitosis_detection/outputs/example_data/{loader_name}/{cell_type}")
        save_path.mkdir(parents=True, exist_ok=True)

        for idx, (img, img2, gt, img_numpy, img2_numpy, crop_np, crop2_np) in enumerate(data_loader):
            img = img * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_img.png")
            img = img2 * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_img2.png")
            img = gt * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_gt.png")

            img = img_numpy * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_img_orig.png")
            img = img2_numpy * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_img2_orig.png")

            img = crop_np * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_crop.png")
            img = crop2_np * 255
            Image.fromarray(img.astype(np.uint8)).save(f"{save_path}/{idx:03d}_crop2.png")
