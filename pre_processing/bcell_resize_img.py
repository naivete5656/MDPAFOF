from pathlib import Path
import argparse

from PIL import Image
import numpy as np
from yaml import parse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--img_dir", help="training dataset's path", default="./datas/source_data", type=str)
    parser.add_argument("--save_path", help="training dataset's path", default="./datas/resized_data", type=str)
    args = parser.parse_args()
    return args


def resize_normal():
    args = parse_args()
    save_path = Path(args.save_path)
    img_dirs = Path(args.img_dir).iterdir()
    for img_dir in img_dirs:
        save_path.joinpath(f"{img_dir.stem}/imgs").mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_dir.glob("*.*"))
        for img_path in img_paths:
            if img_path.parent.stem == "B05f03":
                img = np.array(Image.open(img_path))
                img = (img / 4095) * 255
                img = Image.fromarray(img).convert("L")
            else:
                img = Image.open(img_path).convert("L")

            img = Image.open(img_path).convert("L")

            img = img.resize((1024, 1024))

            save_img_name = save_path.joinpath(f"{img_dir.stem}/imgs/{img_path.stem}.png")
            img.save(save_img_name)


if __name__ == "__main__":
    save_path = Path("./datas/bcell_preprocessed/det_cell")

    data_dir = Path(
        "dataset/MFGTMP_220402110001_jpeg_image_NE_ME_d5_H2B-mCherry_clone44_B04_NE_C04_ME_int5_05sec_3min_3h"
    )

    # img_dirs = data_dir.glob('B04f0*')
    # for img_dir in img_dirs:
    for idx in range(1, 6):
        img_dir = Path(f"{data_dir}/B04f0{idx}")
        save_path.joinpath(f"{img_dir.stem}/imgs").mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_dir.glob("*.*"))
        for img_path in img_paths:
            img = Image.open(img_path).convert("L")

            img = img.resize((1024, 1024))

            save_img_name = save_path.joinpath(f"{img_dir.stem}/imgs/{img_path.stem}.png")
            img.save(save_img_name)
