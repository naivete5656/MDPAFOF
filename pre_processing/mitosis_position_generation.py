from pathlib import Path
import pickle

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


def ctk_annotation2point(img_dir, track_dir, cell_info_list, data_save_path, save_path_vis):
    mit_pos_list = []
    for c_id, s_f, e_f, p_id in cell_info_list:
        if p_id != 0:
            mit_frame = int(s_f - 1)
            track_path = f"{track_dir}/man_track{mit_frame:03d}.tif"
            tra = np.array(Image.open(track_path))
            # mask = tra == p_id
            x_list, y_list = np.where(tra == p_id)
            if x_list.shape[0] > 0:
                mit_pos_list.append([mit_frame, np.mean(x_list), np.mean(y_list)])
    mit_pos_list = np.array(mit_pos_list)

    # frame, x, y
    np.savetxt(str(data_save_path.joinpath("mitosis_pos.txt")), mit_pos_list)

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


def organize_path(data_save_path):
    hm_paths = data_save_path.joinpath("cand_gt").glob("*.png")
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


if __name__ == "__main__":
    base_path = "./datas/original_data/Fluo-N2DL-HeLa/Fluo-N2DL-HeLa"
    for seq in ["01", "02"]:
        img_dir = Path(f"{base_path}/{seq}")
        track_dir = Path(f"{base_path}/{seq}_GT/TRA")
        cell_info_list = np.loadtxt(f"{base_path}/{seq}_GT/TRA/man_track.txt")

        data_save_path = Path(f"/mnt/d/cell_mitosis_detection/datas/Fluo-N2DL-HeLa/{seq}")
        data_save_path.mkdir(parents=True, exist_ok=True)
        save_path_vis = Path(f"/mnt/d/cell_mitosis_detection/output/vis/Fluo-N2DL-HeLa/{seq}")
        save_path_vis.mkdir(parents=True, exist_ok=True)

        # ctk_annotation2point(img_dir, track_dir, cell_info_list, data_save_path, save_path_vis)

        data_save_path.joinpath("cand_gt").mkdir(parents=True, exist_ok=True)
        data_save_path.joinpath("img").mkdir(parents=True, exist_ok=True)

        # pos2hm(img_dir, data_save_path)

        organize_path(data_save_path)

    # base_path = "/mnt/d/cell_mitosis_detection/datas/original_data/Fluo-N2DL-HeLa_test/Fluo-N2DL-HeLa"
    # for seq in ["01", "02"]:
    #     img_dir = Path(f"{base_path}/{seq}")
    #     data_save_path = Path(f"/mnt/d/cell_mitosis_detection/datas/Fluo-N2DL-HeLa_test/{seq}")
    #     data_save_path.joinpath("img").mkdir(parents=True, exist_ok=True)
    #     test_data_save(img_dir, data_save_path)
