from pathlib import Path
import argparse
import copy

from tqdm import tqdm
import numpy as np
import cv2

import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torchvision.models as models

import _init_paths
from models import UNet
from data_loader import MissingCP, MissingSuper
from utils import VisdomClass, hms2pos, create_logger
from evaluation import evaluate_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion):
    model.train()
    losses = 0
    for data in tqdm(dataloader):
        input = data[0].to(device)
        gt = data[1].to(device)
        output = model(input)

        loss = criterion(output, gt)
        losses += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # vis.logging_loss(iteration, loss)
        # if iteration % 20 == 0:
        #     vis.result_show(output, img, gt, 16)
    epoch_loss = losses / (len(dataloader) + 1)
    return epoch_loss


def validate(model, val_loader, criterion):
    model.eval()
    losses = 0
    num_data = 0

    hms = []
    for img, gt in val_loader:
        input = img.to(device)
        gt = gt.to(device)
        output = model(input)
        output = output[:, :, : gt.shape[2], : gt.shape[3]]
        loss = criterion(output, gt)
        losses += loss.item()
        hm = output.detach().cpu().numpy()[:, 0]
        hms.extend(hm)

    num_data += len(val_loader)

    hms = np.array(hms)
    pred_pos = hms2pos(hms)
    pred_pos = pred_pos[:, [2, 0, 1]]

    gt_pos = val_loader.dataset.gt_poses
    gt_pos = gt_pos[:, [2, 0, 1]]

    tp, fp, fn, _, _ = evaluate_detection(pred_pos, gt_pos)
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-20)
    val_loss = losses / num_data
    return f1, val_loss


def test(model, data_loader, save_dir):
    model.eval()
    hms = []

    for img, gt, data_ids in data_loader:
        img = img.to(device)

        outputs = model(img)
        outputs = outputs[:, :, : gt.shape[2], : gt.shape[3]]
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

    np.savetxt(save_path, pred_pos, fmt="%.04f")
    return pred_pos


def visualize(img_paths, results, save_dir):
    save_dir.joinpath("plot").mkdir(parents=True, exist_ok=True)
    for frame, img_path in enumerate(img_paths):
        result_frame = results[results[:, 0] == frame]
        img = np.array(Image.open(img_path[0]))

        plt.figure(figsize=(3, 3))
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


def evaluate(model, test_loader, args):
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    pred_pos = test(model, test_loader, save_dir.joinpath(f"hmpred"))

    pred_pos = pred_pos[:, [2, 0, 1]]

    gt_pos = test_loader.dataset.gt_poses
    gt_pos = gt_pos[:, [2, 0, 1]]
    try:
        # input of evaluation code is [frame, width, height]
        tp, fp, fn, det_res, gt_res = evaluate_detection(pred_pos, gt_pos)
    except:
        print(1)
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-20)

    fps = det_res[det_res[:, 3] == 0]
    fps[:, -1] = 1
    results = np.concatenate([gt_res[:, :4], fps[:, :4]], axis=0)

    # save as [width, height, frame, id]
    np.savetxt(f"{save_dir}/est_result.txt", results[:, [1, 2, 0, 3]], fmt="%d")

    visualize(test_loader.dataset.img_paths, results, save_dir)
    return f1, precision, recall


def main(args):
    weight_path = Path(args.weight_path)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    logger = create_logger(weight_path.parent.joinpath("training_log.log"))

    model = UNet4(2, 1)
    if args.pretrain_path != "":
        model.load_state_dict(torch.load(args.pretrain_path))
    model = model.to(device)

    logger.info(f"{args.img_dir}")
    train_dataset = eval(args.dataloader)(args.img_dir, args.cv_num, args.seed, args.missing)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
    )

    val_dataset = CVValloader(args.img_dir, args.cv_num)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=args.workers)

    test_dataset = CVTestloader(args.img_dir, args.cv_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=args.workers)

    logger.info(f"val sequ = {val_dataset.data_dirs[0].stem}")
    logger.info(f"test sequ = {test_dataset.data_dirs[0].stem}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)

    criterion = nn.MSELoss()

    # vis = VisdomClass()
    # vis.vis_init("Mitosis detection", 16)

    loss_list = []
    val_losses = []
    f1_list = [0]
    for epoch in range(args.epochs):
        epoch_loss = train(model, train_loader, optimizer, criterion)
        loss_list.append(epoch_loss)
        logger.info(f"Epoch :{epoch}/{args.epochs}, loss: {epoch_loss}")

        f1, val_loss = validate(model, val_loader, criterion)
        logger.info(f"Val f1: {f1}")
        if max(f1_list) < f1:
            best_weight = copy.deepcopy(model.state_dict())
        f1_list.append(f1)
        val_losses.append(val_loss)
        if args.save_each_epoch:
            Path(args.weight_path).parent.joinpath("each_epoch").mkdir(parents=True, exist_ok=True)
            save_path = f"{Path(args.weight_path).parent}/each_epoch/{epoch:04d}.pth"
            torch.save(model.state_dict(), save_path)

    logger.info(f"Training complete")
    logger.info(f"Best val f1: {max(f1_list):4f}")

    # load best model weights
    torch.save(model.state_dict(), f"{weight_path.parent}/final.pth")

    torch.save(best_weight, str(weight_path))

    plt.plot(range(args.epochs), loss_list), plt.plot(range(args.epochs), val_losses), plt.savefig(
        str(weight_path.parent.joinpath("loss_curve.png"))
    )
    plt.close()
    np.save(str(weight_path.parent.joinpath("losses.npy")), np.array(loss_list))
    np.save(str(weight_path.parent.joinpath("val_losses.npy")), np.array(val_losses))

    plt.plot(range(args.epochs), f1_list[1:]), plt.savefig(str(weight_path.parent.joinpath("f1_curve.png")))
    plt.close()

    model.load_state_dict(best_weight)
    f1, precision, recall = evaluate(model, test_loader, args)

    logger.info(f"{f1}, {precision}, {recall}")


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "--img_dir", help="training dataset's path", default="./datas/bcell_preprocessed/normal", type=str
    )
    parser.add_argument("--save_path", help="save path", default="./outputs/test", type=str)
    parser.add_argument("--weight_path", help="weigth path", default="./weights/test/best.pth", type=str)
    parser.add_argument("--pretrain_path", help="pretrained path", default="", type=str)
    parser.add_argument("--dataloader", help="data loader", default="MissingCP", type=str)
    parser.add_argument("--cv_num", help="val id", default=0, type=int)
    parser.add_argument("--seed", help="seed", default=0, type=int)
    parser.add_argument("--shot", help="shot", default=5, type=int)
    parser.add_argument("--missing", help="paste", default="0.30", type=str)
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch_size", default=8, type=int)
    parser.add_argument("--epochs", help="epoch", default=100, type=int)
    parser.add_argument("--workers", dest="workers", help="num workers", default=2, type=int)
    parser.add_argument("--save_each_epoch", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
