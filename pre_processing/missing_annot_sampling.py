from pathlib import Path
import copy

import numpy as np
from PIL import Image
import cv2


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


def sampling_shot(cv_num, data_dirs, seed):
    hw_range = 20
    # pop test_dir
    data_dirs.pop(cv_num)
    # pop train_dir
    if cv_num == 3:
        data_dirs.pop(0)
    else:
        data_dirs.pop(cv_num)

    for data_dir in data_dirs:
        gt_path = f"{data_dir}/{data_dir.stem}_mit.txt"
        gt_pos = np.loadtxt(gt_path, comments="%", encoding="shift-jis")
        img_paths = sorted(data_dir.joinpath(f"imgs").glob("*.*"))

        select_gt_pos = gt_pos
        for num_s in range(5, 91, 5):
            data_dir.joinpath(f"missing_{num_s * 0.01:.02f}_new").mkdir(parents=True, exist_ok=True)
            num_shot = int(gt_pos.shape[0] * 0.05)

            select_ids = np.random.choice(select_gt_pos.shape[0], num_shot, replace=False)
            select_gt_pos = np.delete(select_gt_pos, select_ids, axis=0)
            save_path = data_dir.joinpath(f"missing_{num_s * 0.01:.02f}_new/{seed:02d}.txt")
            np.savetxt(str(save_path), select_gt_pos, fmt="%d")

            # save_path = data_dir.joinpath(f"missing_{num_s * 0.01:.02f}_new/{seed:02d}/hms")
            # save_path.mkdir(parents=True, exist_ok=True)

            save_path = data_dir.joinpath(f"missing_{num_s * 0.01:.02f}_new/{seed:02d}/phms")
            save_path.mkdir(parents=True, exist_ok=True)

            img_size = Image.open(img_paths[0]).size

            for frame in np.unique(select_gt_pos[:, 2]):
                mit_pos_frame = select_gt_pos[select_gt_pos[:, 2] == (frame)]
                hm = pos2hm(mit_pos_frame, img_size)

                cv2.imwrite(f"{save_path}/{int(frame):03d}.png", hm * 255)

            # for frame, img_path in enumerate(img_paths[:-1]):
            #     mit_pos_frame = select_gt_pos[select_gt_pos[:, 2] == (frame)]
            #     img_size = Image.open(img_path).size
            #     hm = pos2hm(mit_pos_frame, img_size)

            #     cv2.imwrite(str(save_path.joinpath(f"{save_path}/{int(frame):03d}.png")), hm * 255)

            # for frame in np.unique(select_gt_pos[:, 2]):
            #     mit_pos_frame = select_gt_pos[select_gt_pos[:, 2] == (frame)]
            #     hm = pos2hm(mit_pos_frame, img_size)

            #     cv2.imwrite(str(save_path.joinpath(f"{save_path}/{int(frame):03d}.png")), hm * 255)


DATAPATH = {
    "Fluo-N2DL-HeLa": "ctc_preprocessed",
    "normal": "bcell_preprocessed",
    "det_cell": "bcell_preprocessed",
    "easy_cell": "bcell_preprocessed",
}
if __name__ == "__main__":
    # img_dir = "/mnt/d/main/Mitosis_detection/datas/bcell_preprocessed/normal_type"

    # for cell_type in ["Fluo-N2DL-HeLa", "det_cell", "easy_cell", "normal"]:
    for cell_type in ["Fluo-N2DL-HeLa", "det_cell", "easy_cell", "normal"]:
        img_dir = f"./datas/{DATAPATH[cell_type]}/{cell_type}"
        data_dirs = []
        if Path(img_dir + "_test").exists():
            for cand_dir in Path(img_dir + "_test").iterdir():
                if cand_dir.is_dir():
                    data_dirs.append(cand_dir)
        for cand_dir in Path(img_dir).iterdir():
            if cand_dir.is_dir():
                data_dirs.append(cand_dir)

        for seed in range(5):
            np.random.seed(seed=seed)
            for cv_num in range(4):
                sampling_shot(cv_num, copy.copy(data_dirs), seed)
