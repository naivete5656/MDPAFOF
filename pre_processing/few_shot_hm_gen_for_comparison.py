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
    w_gt, h_gt = int(img_size[0] / stride), int(img_size[1] / stride)
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


def supervised_hm(args):
    base_path = Path(args.annotation_path)
    save_path = Path(args.save_path)
    base_dirs = base_path.iterdir()
    for base_dir in base_dirs:
        if base_dir.is_dir():
            img_paths = sorted(save_path.joinpath("imgs").glob("*.png"))
            anns_path = Path(base_path.joinpath(f"{base_dir.stem}_mit.txt"))

            base_dir.joinpath("hms").mkdir(parents=True, exist_ok=True)

            # x y frame id
            anns = np.loadtxt(str(anns_path), comments="%", encoding="shift-jis")

            for frame, img_path in enumerate(img_paths):
                mit_pos_frame = anns[anns[:, 2] == (frame)]
                img_size = Image.open(img_path).size
                hm = pos2hm(mit_pos_frame, img_size)

                cv2.imwrite(str(save_path.joinpath(f"{base_dir.stem}/hms/{img_path.name}")), hm * 255)
                np.save(str(save_path.joinpath(f"{base_dir.stem}/hms/{img_path.stem}")), hm)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--annotation_path", help="training dataset's path", default="./datas/annotations", type=str)
    parser.add_argument("--save_path", help="training dataset's path", default="./datas/resized_data", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    # supervised_hm(args)
    for cell_type in [
        "ctc_preprocessed/Fluo-N2DL-HeLa",
        "bcell_preprocessed/normal",
        "bcell_preprocessed/det_cell",
        "bcell_preprocessed/easy_cell",
    ]:
        img_dir = f"./datas/{cell_type}"

        data_dirs = []
        if Path(img_dir + "_test").exists():
            for cand_dir in Path(img_dir + "_test").iterdir():
                if cand_dir.is_dir():
                    data_dirs.append(cand_dir)
        for cand_dir in Path(img_dir).iterdir():
            if cand_dir.is_dir():
                data_dirs.append(cand_dir)

        for data_dir in data_dirs:
            if data_dir.is_dir():
                img_paths = sorted(data_dir.joinpath("imgs").glob("*.*"))
                img_path = img_paths[0]
                img_size = Image.open(img_path).size
                for shot in [1, 3, 5, 7]:
                    for seed in range(5):
                        data_dir.joinpath(f"hms{shot}shot/seed{seed}").mkdir(parents=True, exist_ok=True)

                        fs_pos_path = data_dir.joinpath(f"new_fs_{shot}/{seed:02d}.txt")
                        fs_pos = np.loadtxt(str(fs_pos_path), ndmin=2)

                        for frame in np.unique(fs_pos[:, 2]):
                            mit_pos_frame = fs_pos[fs_pos[:, 2] == (frame)]
                            hm = pos2hm(mit_pos_frame, img_size)

                            cv2.imwrite(
                                str(data_dir.joinpath(f"hms{shot}shot/seed{seed}/{int(frame):03d}.png")), hm * 255
                            )
