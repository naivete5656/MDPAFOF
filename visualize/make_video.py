from pathlib import Path
import cv2

def gen_video_original():
    attr_dirs = Path("/mnt/d/main/Mitosis_detection/datas/resized_data").iterdir()
    save_path = Path("/mnt/d/main/Mitosis_detection/datas/vis_video")

    for attr_dir in attr_dirs:
        save_path.joinpath(attr_dir.stem).mkdir(parents=True, exist_ok=True)
        vd_dirs = attr_dir.iterdir()

        for vd_dir in vd_dirs:
            if vd_dir.is_dir():
                img_paths = sorted(vd_dir.joinpath("imgs").glob("*.png"))
                save_name = f"{save_path}/{attr_dir.stem}/{vd_dir.stem}.mp4"

                img = cv2.imread(str(img_paths[0]))
                fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
                video = cv2.VideoWriter(save_name, fourcc, 2, img.shape[:2][::-1])

                for img_path in img_paths:
                    img = cv2.imread(str(img_path))
                    video.write(img)
                video.release()


def gen_anno_video():
    attr_dirs = Path("/mnt/d/main/Mitosis_detection/datas/visualize/mit_cell").iterdir()
    save_path = Path("/mnt/d/main/Mitosis_detection/datas/visualize/anno_video")

    save_path.mkdir(parents=True, exist_ok=True)

    for vd_dir in attr_dirs:
        if vd_dir.is_dir():
            img_paths = sorted(vd_dir.glob("*.png"))
            save_name = f"{save_path}/{vd_dir.stem}.mp4"

            img = cv2.imread(str(img_paths[0]))
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            video = cv2.VideoWriter(save_name, fourcc, 2, img.shape[:2][::-1])

            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                video.write(img)
            video.release()

def gen_anno_video():
    attr_dirs = Path("/mnt/d/main/Mitosis_detection/datas/visualize/mit_cell").iterdir()
    save_path = Path("/mnt/d/main/Mitosis_detection/datas/visualize/anno_video")

    save_path.mkdir(parents=True, exist_ok=True)

    for vd_dir in attr_dirs:
        if vd_dir.is_dir():
            img_paths = sorted(vd_dir.glob("*.png"))
            save_name = f"{save_path}/{vd_dir.stem}.mp4"

            img = cv2.imread(str(img_paths[0]))
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            video = cv2.VideoWriter(save_name, fourcc, 2, img.shape[:2][::-1])

            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                video.write(img)
            video.release()

def gen_video():
    vd_dir = Path("/mnt/d/main/Mitosis_detection/outputs/cpwofli/plot")
    save_path = Path("/mnt/d/main/Mitosis_detection/outputs/cpwofli")

    save_path.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(vd_dir.glob("*.png"))
    save_name = f"{save_path}/{vd_dir.stem}.mp4"

    img = cv2.imread(str(img_paths[0]))
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    video = cv2.VideoWriter(save_name, fourcc, 2, img.shape[:2][::-1])

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        video.write(img)
    video.release() 

    
if __name__=='__main__':
    gen_video()
    
