import torch

def hm2pos(hms, peak_th=0.3, min_dist=11):
    mppool = torch.nn.MaxPool2d(min_dist, 1, min_dist // 2)

    maxm = mppool(hms)
    maxm = torch.eq(maxm, hms).float()
    heatmaps = hms * maxm

    shifts_x = torch.arange(0, hms.shape[-1], step=1, dtype=torch.float32)
    shifts_y = torch.arange(0, hms.shape[-2], step=1, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    scores = heatmaps.view(hms.shape[0], -1)
    scores, pos_inds = scores.topk(500, dim=1)
    peaks = []
    for score, pos_ind in zip(scores, pos_inds):
        select_ind = (score > (peak_th)).nonzero()
        pos_ind = pos_ind[select_ind][:, 0]
        # torch.cat([locations[pos_ind], torch.zeros((33, 1))], dim=1).shape
        peaks.append(locations[pos_ind])
    return peaks

def hms2pos(hms, peak_th=0.3, min_dist=11):
    '''
    hms = (x, y, t)
    hms = (1, 1, x, y, t)
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

    select_ind = (score > (peak_th)).nonzero()
    pos_ind_th = pos_ind[select_ind][:, 0]
    peaks = locations[pos_ind_th].numpy()
    peaks[:, :2] = peaks[:, :2] * 4
    return peaks