from pathlib import Path
import argparse

import torch
import numpy as np


def hm2pos(save_path, hms, celltype, peak_th=0.3, min_dist=11):
    '''
    hms = (B, C, W, H)
    '''
    hms = torch.from_numpy(hms).unsqueeze(0).unsqueeze(0).float()
    mppool = torch.nn.MaxPool3d(min_dist, 1, min_dist // 2)

    maxm = mppool(hms)
    maxm = torch.eq(maxm, hms).float()
    heatmaps = hms * maxm

    shifts_x = torch.arange(0, hms.shape[-1], step=1, dtype=torch.float32)
    shifts_y = torch.arange(0, hms.shape[-2], step=1, dtype=torch.float32)
    shifts_z = torch.arange(0, hms.shape[-3], step=1, dtype=torch.float32)

    shift_z, shift_y, shift_x = torch.meshgrid(shifts_z, shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shift_z = shift_z.reshape(-1)
    locations = torch.stack((shift_x, shift_y, shift_z), dim=1)

    scores = heatmaps.view(hms.shape[0], -1)
    scores, pos_inds = scores.topk(500, dim=1)
    score, pos_ind = scores[0], pos_inds[0]

    select_ind = (score > peak_th).nonzero()
    pos_ind_th = pos_ind[select_ind][:, 0]
    peaks = locations[pos_ind_th].numpy()
    peaks[:, :2] = peaks[:, :2] * 4
    np.savetxt(str(save_path.joinpath(f"{celltype}.txt")), peaks, fmt='%.04f')

    return peaks

def aggregate_result(pred_dir, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    pred_paths = sorted(pred_dir.glob("*.npy"))

    pred_vid = []
    for pred_path in pred_paths:
        pred = np.load(pred_path)
        pred_vid.append(pred)
    pred_vid = np.stack(pred_vid)

    # x, y, t
    peaks = hm2pos(save_path, pred_vid, pred_dir.stem, 0.3, 11)
    return peaks