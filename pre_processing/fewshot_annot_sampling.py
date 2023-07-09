from pathlib import Path
import copy

import numpy as np
from PIL import Image


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
        gt_path = f"{data_dir}_mit.txt"
        gt_pos = np.loadtxt(gt_path, comments="%", encoding="shift-jis")
        img_paths = sorted(data_dir.joinpath(f"img").glob("*.*"))

        num_shot = 5
        data_dir.joinpath(f"fs_{num_shot}/crop{seed:02d}").mkdir(parents=True, exist_ok=True)
        select_ids = np.random.choice(gt_pos.shape[0], num_shot, replace=False)
        select_gt_pos = gt_pos[select_ids].astype(np.int32)
        save_path = data_dir.joinpath(f"fs_{num_shot}/{seed:02d}.txt")
        np.savetxt(str(save_path), select_gt_pos, fmt="%d")

        for idx, (x, y, t, _) in enumerate(select_gt_pos):
            img_path = img_paths[t - 1]
            img = np.array(Image.open(img_path).convert("L"))
            img = np.pad(img, hw_range, "symmetric")
            crop_img = img[y : y + 2 * hw_range, x : x + 2 * hw_range]
            img_path = img_paths[t]
            img = np.array(Image.open(img_path).convert("L"))
            img = np.pad(img, hw_range, "symmetric")
            crop_img2 = img[y : y + 2 * hw_range, x : x + 2 * hw_range]

            save_path = data_dir.joinpath(f"fs_{num_shot}/crop{seed:02d}/{idx:02d}_1.png")
            Image.fromarray(crop_img).save(save_path)

            save_path = data_dir.joinpath(f"fs_{num_shot}/crop{seed:02d}/{idx:02d}_2.png")
            Image.fromarray(crop_img2).save(save_path)


def sampling_fix(cv_num, data_dirs, seed):
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

        shot_path = data_dir.joinpath(f"fs_5/{seed:02d}.txt")
        five_shot_pos = np.loadtxt(str(shot_path))

        for cell_id in five_shot_pos[:, -1]:
            gt_pos = gt_pos[gt_pos[:, -1] != cell_id]

        data_dir.joinpath(f"fs_5/crop{seed:02d}").mkdir(parents=True, exist_ok=True)
        save_path = data_dir.joinpath(f"fs_5/{seed:02d}.txt")
        np.savetxt(str(save_path), five_shot_pos, fmt="%d")
        vis_cropped_img(data_dir, five_shot_pos, 5)

        for num_shot in [1, 3]:
            select_ids = np.random.choice(five_shot_pos.shape[0], num_shot, replace=False)
            select_gt_pos = five_shot_pos[select_ids].astype(np.int32)

            data_dir.joinpath(f"fs_{num_shot}/crop{seed:02d}").mkdir(parents=True, exist_ok=True)
            save_path = data_dir.joinpath(f"fs_{num_shot}/{seed:02d}.txt")
            np.savetxt(str(save_path), select_gt_pos, fmt="%d")

            vis_cropped_img(data_dir, select_gt_pos, num_shot)

        for num_shot in [7]:
            select_ids = np.random.choice(gt_pos.shape[0], num_shot - 5, replace=False)
            select_gt_pos = gt_pos[select_ids].astype(np.int32)
            select_gt_pos = np.concatenate([five_shot_pos, select_gt_pos], axis=0)

            data_dir.joinpath(f"fs_{num_shot}/crop{seed:02d}").mkdir(parents=True, exist_ok=True)
            save_path = data_dir.joinpath(f"fs_{num_shot}/{seed:02d}.txt")
            np.savetxt(str(save_path), select_gt_pos, fmt="%d")

            vis_cropped_img(data_dir, select_gt_pos, num_shot)


def vis_cropped_img(data_dir, select_gt_pos, num_shot, hw_range=40):
    img_paths = sorted(data_dir.joinpath(f"imgs").glob("*.*"))
    select_gt_pos = select_gt_pos.astype(np.int32)
    for idx, (x, y, t, _) in enumerate(select_gt_pos):
        img_path = img_paths[t - 1]
        img = np.array(Image.open(img_path).convert("L"))
        img = np.pad(img, hw_range, "symmetric")
        crop_img = img[y : y + 2 * hw_range, x : x + 2 * hw_range]
        img_path = img_paths[t]
        img = np.array(Image.open(img_path).convert("L"))
        img = np.pad(img, hw_range, "symmetric")
        crop_img2 = img[y : y + 2 * hw_range, x : x + 2 * hw_range]

        save_path = data_dir.joinpath(f"new_fs_{num_shot}/crop{seed:02d}/{idx:02d}_1.png")
        Image.fromarray(crop_img).save(save_path)

        save_path = data_dir.joinpath(f"new_fs_{num_shot}/crop{seed:02d}/{idx:02d}_2.png")
        Image.fromarray(crop_img2).save(save_path)


DATAPATH = {
    "Fluo-N2DL-HeLa": "ctc_preprocessed",
    "normal": "bcell_preprocessed",
    "det_cell": "bcell_preprocessed",
    "easy_cell": "bcell_preprocessed",
}

if __name__ == "__main__":
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
                #             sampling_shot(cv_num, copy.copy(data_dirs), seed)
                sampling_fix(cv_num, copy.copy(data_dirs), seed)
