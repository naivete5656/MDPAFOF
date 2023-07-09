from pathlib import Path
import pickle
import argparse

from PIL import Image
import numpy as np
import cv2
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt


def local_maximum(img, threshold=50, dist=2):
    assert len(img.shape) == 2
    data = np.zeros((0, 2))
    x = peak_local_max(img, threshold_abs=threshold, min_distance=dist)
    peak_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for j in range(x.shape[0]):
        peak_img[x[j, 0], x[j, 1]] = 255
    labels, _, _, center = cv2.connectedComponentsWithStats(peak_img)
    for j in range(1, labels):
        data = np.append(data, [[center[j, 0], center[j, 1]]], axis=0).astype(int)
    return data


def ctc_annotation2point(img_dir, track_dir, cell_info_list, data_save_path, save_path_vis, cell_type, seq):
    mit_pos_list = []
    added_id = []
    for c_id, s_f, e_f, p_id in cell_info_list:
        if p_id != 0:
            mit_frame = int(s_f - 1)
            track_path = f"{track_dir}/man_track{mit_frame:03d}.tif"
            tra = np.array(Image.open(track_path))
            # mask = tra == p_id
            x_list, y_list = np.where(tra == p_id)
            if (x_list.shape[0] > 0) and (p_id not in added_id):
                mit_pos_list.append([np.mean(y_list), np.mean(x_list), mit_frame])
            added_id.append(p_id)
    mit_pos_list = np.array(mit_pos_list)

    if cell_type == "Fluo-N2DH-SIM+":
        mit_pos_list[:, 2] -= 1
    else:
        mit_pos_list[:, 2] += 1

    # x, y, t
    np.savetxt(str(data_save_path.joinpath(f"{seq}_mit.txt")), mit_pos_list)

    img_paths = img_dir.glob("*.tif")
    norm_max = 0
    norm_min = 1e5
    for frame, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path), -1)
        norm_max = max(img.max(), norm_max)
        norm_min = min(img.min(), norm_min)
        img = (((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        mit_pos_frame = mit_pos_list[mit_pos_list[:, 0] == frame]
        for frame, x, y in mit_pos_frame:
            img = cv2.circle(img, (int(y), int(x)), 7, (255, 0, 0), thickness=3, lineType=cv2.LINE_8)
        save_vis_path = save_path_vis.joinpath(f"{int(frame):03d}.tif")
        cv2.imwrite(str(save_vis_path), img)
    with data_save_path.joinpath("norm_value.pkl").open("wb") as f:
        pickle.dump([norm_max, norm_min], f)


def gaussian(x, y, t, x_gt, y_gt, t_gt, sigma=6):
    return (
        1
        / np.sqrt(2 * np.pi * sigma ^ 2)
        * np.exp((-(x - x_gt) ^ 2 - (y - y_gt) ^ 2 - 5 * (t - t_gt) ^ 2) / 2 * sigma ^ 2)
    )


def pos2hm(img_dir, data_save_path, stride=4, t_range=1, sigma=3):
    mit_pos_list = np.loadtxt(f"{data_save_path}/mitosis_pos.txt")
    with data_save_path.joinpath("norm_value.pkl").open("rb") as f:
        norm_max, norm_min = pickle.load(f)

    img_paths = sorted(img_dir.glob("*.tif"))
    for frame, img_path in enumerate(img_paths):
        img = np.array(Image.open(img_path))
        img = (img - norm_min) / (norm_max - norm_min) * 255
        Image.fromarray(img.astype(np.uint8)).save(data_save_path.joinpath(f"img/{img_path.name}"))

        h, w = img.shape[:2]
        h_gt, w_gt = int(h / stride), int(w / stride)
        gt = np.zeros((h_gt, w_gt))
        size = sigma * 5 + 1
        gt = np.pad(gt, (size // 2, size // 2 + 1))

        mit_pos_frame = mit_pos_list[(mit_pos_list[:, 0] > frame - t_range) & (mit_pos_list[:, 0] < frame + t_range)]

        if mit_pos_frame.shape[0] > 0:
            for mit_frame, y, x in mit_pos_frame:
                y_scale = y / stride
                y_dec, y_int = np.modf(y_scale)
                x_scale = x / stride
                x_dec, x_int = np.modf(x_scale)

                t_dif = frame - mit_frame
                x = np.arange(0, size, 1, np.float32)
                shift_y, shift_x = np.meshgrid(x, x)
                x0 = y0 = size // 2
                y0 += y_dec
                x0 += x_dec

                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(-((shift_x - x0) ** 2 + (shift_y - y0) ** 2 + 5 * (t_dif**2)) / (2 * sigma**2))

                top = int(max(0, y_int))
                bot = int(min(h_gt + size, y_int + size))
                left = int(max(0, x_int))
                right = int(min(w_gt + size, x_int + size))
                gt[top:bot, left:right] = np.maximum(gt[top:bot, left:right], g)
        gt = gt[size // 2 : -(size // 2 + 1), size // 2 : -(size // 2 + 1)]
        cv2.imwrite(str(data_save_path.joinpath(f"cand_gt/t{frame:03d}.png")), gt * 255)


def pos22inputhm(img_dir, mit_pos_list, data_save_path, stride=4, t_range=1, sigma=3):
    img_paths = sorted(img_dir.glob("*.tif"))
    for frame, img_path in enumerate(img_paths[1:], 1):
        img_size = Image.open(img_path).size
        mit_pos_frame = mit_pos_list[mit_pos_list[:, 0] == (frame)]

        window_size = sigma * 5 + 1
        h_gt, w_gt = int(img_size[1] / stride), int(img_size[0] / stride)
        gt = np.zeros((h_gt, w_gt))
        if mit_pos_frame.shape[0] > 0:
            gt = np.pad(gt, (window_size // 2, window_size // 2 + 1))

            for t, y, x in mit_pos_frame:
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
        cv2.imwrite(str(data_save_path.joinpath(f"hms/t{frame:03d}.png")), gt * 255)


def organize_path(data_save_path):
    hm_paths = data_save_path.joinpath("hms").glob("*.png")
    data_type = {"pos": [], "neg": []}
    peak_pos = {}
    for hm_path in hm_paths:
        hm = np.array(Image.open(hm_path))
        m_value = hm.max()
        if m_value == 0:
            data_type["neg"].append(int(hm_path.stem[1:]))
        else:
            data_type["pos"].append(int(hm_path.stem[1:]))
            peaks = local_maximum(hm)
            peak_pos[hm_path.stem] = peaks
    with data_save_path.joinpath("data_type.pkl").open("wb") as f:
        pickle.dump(data_type, f)

    with data_save_path.joinpath("peak_pos.pkl").open("wb") as f:
        pickle.dump(peak_pos, f)


def test_data_save(img_dir, data_save_path):
    img_paths = sorted(img_dir.glob("*.tif"))
    norm_max = 0
    norm_min = 1e5
    for img_path in img_paths:
        img = cv2.imread(str(img_path), -1)
        norm_max = max(img.max(), norm_max)
        norm_min = min(img.min(), norm_min)

    for img_path in img_paths:
        img = np.array(Image.open(img_path))
        img = (img - norm_min) / (norm_max - norm_min) * 255
        Image.fromarray(img.astype(np.uint8)).save(data_save_path.joinpath(f"img/{img_path.name}"))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--img_dir", help="training dataset's path", default="./datas/original_data", type=str)
    parser.add_argument("--save_path", help="training dataset's path", default="./datas/ctc_preprocessed", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    for cell_type in ["Fluo-N2DL-HeLa"]:
        base_path = f"{args.img_dir}/{cell_type}"
        for seq in ["01", "02"]:
            img_dir = Path(f"{base_path}/{seq}")
            track_dir = Path(f"{base_path}/{seq}_GT/TRA")
            cell_info_list = np.loadtxt(f"{base_path}/{seq}_GT/TRA/man_track.txt")

            data_save_path = Path(f"{args.save_path}/{cell_type}/{seq}")
            data_save_path.mkdir(parents=True, exist_ok=True)
            save_path_vis = Path(f"./outputs/visualize/{cell_type}/{seq}")
            save_path_vis.mkdir(parents=True, exist_ok=True)

            ctc_annotation2point(img_dir, track_dir, cell_info_list, data_save_path, save_path_vis, cell_type, seq)

            data_save_path.joinpath("hms").mkdir(parents=True, exist_ok=True)
            data_save_path.joinpath("img").mkdir(parents=True, exist_ok=True)

            mit_pos_list = np.loadtxt(f"{data_save_path}/{seq}_mit.txt")
            mit_pos_list = mit_pos_list[:, [2, 1, 0]]
            pos22inputhm(img_dir, mit_pos_list, data_save_path)
            # pos2hm(img_dir, data_save_path)

        # organize_path(data_save_path)

        base_path = f"{args.img_dir}/{cell_type}_test"
        for seq in ["01", "02"]:
            img_dir = Path(f"{base_path}/{seq}")
            data_save_path = Path(f"{args.save_path}/{cell_type}_test/{seq}")
            data_save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
            test_data_save(img_dir, data_save_path)

    for cell_type in ["Fluo-N2DH-SIM+", "Fluo-N2DL-HeLa"]:
        base_path = f"{args.save_path}/{cell_type}_test"
        for seq in ["01", "02"]:
            img_dir = Path(f"{base_path}/{seq}/img")
            data_save_path = Path(f"{base_path}/{seq}")
            data_save_path.joinpath("hms").mkdir(parents=True, exist_ok=True)

            mit_pos_list = np.loadtxt(f"{base_path}/{seq}_mit.txt", comments="%", encoding="shift-jis")

            pos22inputhm(img_dir, mit_pos_list[:, [2, 1, 0]], data_save_path)
