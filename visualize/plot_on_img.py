from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2


if __name__=='__main__':
    img_paths = sorted(Path("/mnt/d/main/cell_mitosis_detection/datas/resized_data/B02f02").glob("*.png"))
    pred_pos = np.loadtxt("/mnt/d/main/cell_mitosis_detection/output/cand_test_pred_pos/B02f02/mit_pred_test.txt")

    for frame, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))

        mit_pos = pred_pos[pred_pos[:, 2] == frame]
        for pos in mit_pos:
            cv2.circle(img, (int(pos[0]) * 4, int(pos[1]) * 4), 10, color=(255, 0, 0), thickness=-1)

        cv2.imwrite()
        print(1)
