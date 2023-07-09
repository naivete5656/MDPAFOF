from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from EvaluationMetric import evaluate_detection


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "--gt_dir",
        help="training dataset's path",
        default="./datas/resized_data",
        type=str,
    )
    parser.add_argument(
        "--pred_path",
        help="training dataset's path",
        default="./outputs/2in1out/peaks",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    gt_paths = sorted(Path(args.gt_dir).glob("*_mitosis.txt"))
    pred_paths = sorted(Path(args.pred_path).glob("*/**/"))
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        # if pred_path.stem != 'B02f24':
        gt_pos = np.loadtxt(gt_path, comments="%", encoding="shift-jis")
        gt_pos = gt_pos[:, [2, 0, 1]]

        results = []
        for th in range(10):
            pred_pos = np.loadtxt(str(pred_path.joinpath(f"{th:02d}.txt")), ndmin=2)
            if pred_pos.shape[0] > 0:
                pred_pos = pred_pos[:, [2, 0, 1]]
                TP, FP, FN, detection_result, gt_labels = evaluate_detection(pred_pos, gt_pos)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-20)
                results.append([th * 0.1, precision, recall, f1])
            else:
                results.append([th * 0.1, 0, 0, 0])

        results = np.array(results)
        results = np.vstack([results, results[np.argmax(results[:, -1])]])
        np.savetxt(str(pred_path) + ".txt", results, fmt="%.4f", comments="# th, precision, recall, f1")
