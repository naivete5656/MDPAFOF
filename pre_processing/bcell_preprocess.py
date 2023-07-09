from pathlib import Path
import pickle
import argparse

import numpy as np
import cv2
from PIL import Image


def gaussian(x, y, t, x_gt, y_gt, t_gt, sigma=6):
    return (
        1
        / np.sqrt(2 * np.pi * sigma ^ 2)
        * np.exp((-(x - x_gt) ^ 2 - (y - y_gt) ^ 2 - 5 * (t - t_gt) ^ 2) / 2 * sigma ^ 2)
    )


def pos2hm(mit_pos_frame, img_size, stride=4, t_range=1, sigma=3):
    window_size = sigma * 5 + 1
    h_gt, w_gt = int(img_size[0] / stride), int(img_size[1] / stride)
    gt = np.zeros((h_gt, w_gt))
    if mit_pos_frame.shape[0] > 0:
        gt = np.pad(gt, (window_size // 2, window_size // 2 + 1))

        for x, y in mit_pos_frame[:, :2]:
            y_scale = y / stride
            y_dec, y_int = np.modf(y_scale)
            x_scale = x / stride
            x_dec, x_int = np.modf(x_scale)

            x = np.arange(0, window_size, 1, np.float32)
            shift_y, shift_x = np.meshgrid(x, x)
            x0 = y0 = window_size // 2
            y0 += y_dec
            x0 += x_dec

            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(-((shift_x - x0) ** 2 + (shift_y - y0) ** 2) / (2 * sigma**2))

            top = int(max(0, y_int))
            bot = int(min(h_gt + window_size, y_int + window_size))
            left = int(max(0, x_int))
            right = int(min(w_gt + window_size, x_int + window_size))
            gt[top:bot, left:right] = np.maximum(gt[top:bot, left:right], g)
        gt = gt[window_size // 2 : -(window_size // 2 + 1), window_size // 2 : -(window_size // 2 + 1)]

    return gt


if __name__ == "__main__":
    base_path = Path("./datas/bcell_preprocessed/det_cell")
    for img_dir in base_path.iterdir():
        anns_path = f"{img_dir}/{img_dir.stem}_mit.txt"
        anns = np.loadtxt(anns_path, comments="%")

        img_paths = sorted(Path(f"{img_dir}/imgs").glob("*.png"))

        img_dir.joinpath("hms").mkdir(parents=True, exist_ok=True)

        for frame, img_path in enumerate(img_paths[1:], 1):
            mit_pos_frame = anns[anns[:, 2] == (frame)]
            img_size = Image.open(img_path).size
            hm = pos2hm(mit_pos_frame, img_size)

            save_path = f"{img_dir}/hms/{img_path.name}"
            Image.fromarray((hm * 255).astype(np.uint8)).save(save_path)
