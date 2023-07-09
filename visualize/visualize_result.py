from pathlib import Path
import argparse

import numpy as np
import cv2
import matplotlib as mpl
mpl.use('pdf')  # or whatever other backend that you want
import matplotlib.pyplot as plt

import torch

import sys 
sys.path.append('/mnt/d/main/Mitosis_detection')

from models import UNet4
from data_loader import *
from utils import hms2pos
from evaluation import evaluate_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, data_loader, save_dir):
    model.eval()
    hms = []

    for img, gt, data_ids in data_loader:
        img = img.to(device)

        outputs = model(img)
        outputs = outputs[:, :, :gt.shape[2], :gt.shape[3]]
        outputs = outputs.detach().cpu().numpy()
        hms.extend(outputs[:, 0])
        
        for data_id, output in zip(data_ids, outputs):
            out = output.clip(0, 1)[0]
            img_path = data_loader.dataset.img_paths[data_id]
            img_name = img_path[1].stem

            save_path = f"{save_dir}/{img_name}.npy"
            Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
            np.save(save_path, out)

            save_path = f"{save_dir}/{img_name}.png"
            Path(f"{save_dir}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, out * 255)
    hms = np.array(hms)    
    pred_pos = hms2pos(hms)
    save_path = f"{save_dir.parent.parent}/peak2/{save_dir.stem}.txt"
    Path(f"{save_dir.parent.parent}/peak2").mkdir(parents=True, exist_ok=True)

    np.savetxt(save_path, pred_pos, fmt='%.04f')
    return pred_pos

def visualize(img_paths, results, save_dir):
    save_dir.joinpath('plot').mkdir(parents=True, exist_ok=True)
    for frame, img_path in enumerate(img_paths):
        result_frame = results[results[:, 0] == frame]
        img = np.array(Image.open(img_path[0]))

        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap='gray')
        true_positive = result_frame[result_frame[:, 3] == 2]
        plt.plot(true_positive[:, 1], true_positive[:, 2], "rx", label="true positive")
        false_positive = result_frame[result_frame[:, 3] == 1]
        plt.plot(false_positive[:, 1], false_positive[:, 2], "bx", label="false positive")
        false_negative = result_frame[result_frame[:, 3] == 0]
        plt.plot(false_negative[:, 1], false_negative[:, 2], "gx", label="false negative")
        plt.legend(bbox_to_anchor=(0, 1), loc="upper left", fontsize=4, ncol=4)
        plt.axis("off")
        plt.legend()
        plt.savefig(f'{save_dir}/plot/{frame:02d}.png', bbox_inches='tight', pad_inches=0, transparent=True)


def evaluate(model, test_loader, args):
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    pred_pos = test(model, test_loader, save_dir.joinpath(f"hmpred"))

    pred_pos = pred_pos[:, [2, 0, 1]]

    gt_pos = test_loader.dataset.gt_poses
    gt_pos = gt_pos[:, [2, 0, 1]]
    try:
        tp, fp, fn, det_res , gt_res = evaluate_detection(pred_pos, gt_pos)
    except:
        print(1)
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-20)

    fps = det_res[det_res[:, 3] == 0]
    fps[:, -1] = 1
    results = np.concatenate([gt_res[:, :4], fps[:, :4]], axis=0)
    print(f'{f1}, {precision}, {recall}')

    visualize(test_loader.dataset.img_paths, results, save_dir)
    return f1, precision, recall


def main(args):

    model = UNet4(2, 1)
    if args.pretrain_path != "":
        model.load_state_dict(torch.load(args.pretrain_path))
    model = model.to(device)

    test_dataset = CVTestloader(args.img_dir, args.cv_num, args.seed, args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    f1, precision, recall = evaluate(model, test_loader, args)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    # parser.add_argument("--img_dir", help="training dataset's path",default="./datas/ctc_preprocessed/Fluo-N2DH-SIM+", type=str)
    parser.add_argument("--img_dir", help="training dataset's path",default="./datas/bcell_preprocessed/normal_type", type=str)
    parser.add_argument("--save_path", help="save path",default="./outputs/cpwofli", type=str)
    parser.add_argument("--pretrain_path", help="pretrained path",default="/mnt/d/main/Mitosis_detection/weights/copy_pastewoflip/1/best.pth", type=str)
    parser.add_argument("--dataloader", help="data loader",default="CVCPLoader", type=str)
    parser.add_argument("--cv_num", help="val id", default=1, type=int)
    parser.add_argument("--seed", help="seed", default=0, type=int)
    parser.add_argument("--shot", help="shot", default=5, type=int)
    parser.add_argument("--paste", help="shot", default=5, type=int)
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch_size", default=4, type=int)
    parser.add_argument("--epochs", help="epoch", default=100, type=int)
    parser.add_argument("--workers", dest="workers", help="num workers", default=2, type=int)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
            
    