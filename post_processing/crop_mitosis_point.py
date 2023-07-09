from genericpath import exists
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop_mitosi_point(img_dir, source_dir, mit_points, save_dir, t_range=10, hw_range=30):
    save_dir.joinpath("overview").mkdir(parents=True, exist_ok=True)
    save_dir.joinpath("individual_imgs").mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(img_dir).glob("*.png"))
    source_paths = sorted(Path(source_dir).glob("*.*"))

    img = np.array(Image.open(img_paths[0]))
    source_img = np.array(Image.open(source_paths[0]))
    scale = source_img.shape[0] / img.shape[0] 

    mit_points = mit_points.astype(np.int64)
    
    #　画像の端は除くcode
    mit_points = mit_points[(mit_points[:, 0] >= hw_range) & (mit_points[:, 0] <= img.shape[0] - hw_range) & (mit_points[:, 1] >= hw_range) & (mit_points[:, 1] <= img.shape[1] - hw_range)]

    for h, w, t in mit_points:
        cropped_imgs = []
        for t_idx in range(t - t_range, t + t_range):
            save_dir.joinpath(f"individual_imgs/frame{t}_w{w}_h{h}").mkdir(parents=True, exist_ok=True)
            if (t_idx < 0) or (t_idx >= len(img_paths)):
                crop_img = np.full((round(hw_range * scale) * 2, round(hw_range * scale) * 2), 255, dtype=np.uint8)
            else:
                img = np.array(Image.open(source_paths[t_idx]).convert('L'))
                w_source = round(w * scale)
                h_source = round(h * scale)
                img = np.pad(img, round(hw_range * scale), 'constant',constant_values=255)
                left, right, top, bottom = w_source, w_source + 2*round(hw_range * scale), h_source, h_source + 2*round(hw_range * scale)
                crop_img = img[left:right, top:bottom]
                crop_img_norm = (crop_img - crop_img.min()) / (crop_img.max() - crop_img.min()) * 255
                crop_img_norm = crop_img_norm.astype(np.uint8)
                save_path = save_dir.joinpath(f"individual_imgs/frame{t}_w{w_source}_h{h_source}/{t_idx}-{t}_{t_idx-t}.png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop_img_norm).save(save_path)
            cropped_imgs.append(crop_img)
            cropped_imgs.append(np.full((crop_img.shape[0], 1), 255, dtype=np.uint8))
        mit_img = np.hstack(cropped_imgs)
        save_path = save_dir.joinpath(f"overview/frame{t}_w{w}_h{h}.png")
        Image.fromarray(mit_img).save(save_path)

        
