from pathlib import Path
import argparse

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch

import _init_paths
from models import UNet
from data_loader import *
from utils import VisdomClass, hms2pos, create_logger
from evaluation import evaluate_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, data_loader, save_dir):
    model.eval()
    hms = []

    for imgs, gts, data_ids in data_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        outputs = outputs[:, :, : gts.shape[2], : gts.shape[3]]
        outputs = outputs.detach().cpu().numpy()
        hms.extend(outputs[:, 0])

        for data_id, output, img in zip(data_ids, outputs, imgs):
            out = output.clip(0, 1)[0]
            img_path = data_loader.dataset.img_paths[data_id]
            img_name = img_path[1].stem

            img_np = np.array(Image.fromarray(img[0].cpu().numpy()).resize(out.shape[::-1]))
            img2_np = np.array(Image.fromarray(img[1].cpu().numpy()).resize(out.shape[::-1]))
            # out = np.array(Image.fromarray(out))
            save_path = f"{save_dir}/{data_id}_img1.png"
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            cv2.imwrite(save_path, img_np.clip(0, 0.5) * 512)

            save_path = f"{save_dir}/{data_id}_img2.png"
            img2_np = (img2_np - img2_np.min()) / (img2_np.max() - img2_np.min())
            cv2.imwrite(save_path, img2_np.clip(0, 0.5) * 512)

            save_path = f"{save_dir}/{data_id}_hm.png"
            cv2.imwrite(save_path, out * 255)

            res = np.hstack([img_np, img2_np, out])
            save_path = f"{save_dir}/{img_name}.png"
            Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, res * 255)
    hms = np.array(hms)
    pred_pos = hms2pos(hms)
    save_path = f"{save_dir.parent.parent}/peak/{save_dir.stem}.txt"
    Path(f"{save_dir.parent.parent}/peak").mkdir(parents=True, exist_ok=True)

    np.savetxt(save_path, pred_pos, fmt="%.04f")
    return pred_pos


def visualize(img_paths, results, save_dir):
    results[:, 1:3] = results[:, 1:3] / 4
    save_dir.joinpath("plot").mkdir(parents=True, exist_ok=True)
    for frame, img_path in enumerate(img_paths):
        result_frame = results[(results[:, 0] >= frame - 1) & (results[:, 0] <= frame + 1)]
        img = np.array(Image.open(img_path))

        plt.imshow(img, cmap="gray")
        true_positive = result_frame[result_frame[:, 3] == 2]
        plt.plot(true_positive[:, 1], true_positive[:, 2], "rx", label="true positive")
        false_positive = result_frame[result_frame[:, 3] == 1]
        plt.plot(false_positive[:, 1], false_positive[:, 2], "bx", label="false positive")
        false_negative = result_frame[result_frame[:, 3] == 0]
        plt.plot(false_negative[:, 1], false_negative[:, 2], "gx", label="false negative")
        plt.legend(bbox_to_anchor=(0, 1), loc="upper left", fontsize=4, ncol=4)
        plt.axis("off")
        plt.savefig(f"{save_dir}/plot/{frame:02d}.png", bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close()


def evaluate(model, data_loader, args, mode):
    save_dir = Path(f"{args.save_path}/{mode}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    pred_pos = test(model, data_loader, save_dir.joinpath(f"hmpred"))
    pred_pos = pred_pos[:, [2, 0, 1]]

    gt_pos = data_loader.dataset.gt_poses
    gt_pos = gt_pos[:, [2, 0, 1]]
    try:
        tp, fp, fn, det_res, gt_res = evaluate_detection(pred_pos, gt_pos)
    except:
        print(1)
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-20)

    fps = det_res[det_res[:, 3] == 0]
    fps[:, -1] = 1
    results = np.concatenate([gt_res[:, :4], fps[:, :4]], axis=0)
    np.savetxt(f"{save_dir}/gt_res.txt", results[:, [1, 2, 0, 3]], fmt="%d")
    np.savetxt(f"{save_dir}/pred_result.txt", pred_pos, fmt="%d")

    visualize(sorted(save_dir.joinpath(f"hmpred").glob("*.png")), results, save_dir)
    return f1, precision, recall


def main(args):
    model = UNet(2, 1)
    try:
        model.load_state_dict(torch.load(args.weight_path))
    except FileNotFoundError:
        weight_path = f"{args.weight_path[:-8]}final.pth"
        model.load_state_dict(torch.load(weight_path))

    model = model.to(device)

    # val_dataset = CVValTestloader(args.img_dir, args.cv_num)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # f1, precision, recall = evaluate(model, val_loader, args, 'val')
    # print(f'{f1}, {precision}, {recall}')

    test_dataset = CVTestloader(args.img_dir, args.cv_num)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    f1, precision, recall = evaluate(model, test_loader, args, "test")
    print(f"{f1}, {precision}, {recall}")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    # parser.add_argument("--img_dir", help="training dataset's path",default="./datas/ctc_preprocessed/Fluo-N2DL-HeLa", type=str)
    parser.add_argument(
        "--img_dir", help="training dataset's path", default="./datas/bcell_preprocessed/det_cell", type=str
    )
    parser.add_argument("--save_path", help="save path", default="./outputs/test", type=str)

    parser.add_argument(
        "--weight_path",
        help="weigth path",
        default="./weights/bcelldet_cell/partialsup/shot1/seed1/0/best.pth",
        type=str,
    )

    parser.add_argument("--cv_num", help="val id", default=1, type=int)
    parser.add_argument("--seed", help="seed", default=0, type=int)
    parser.add_argument("--shot", help="shot", default=5, type=int)
    parser.add_argument("--paste", help="shot", default=5, type=int)
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch_size", default=8, type=int)
    parser.add_argument("--workers", dest="workers", help="num workers", default=2, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
