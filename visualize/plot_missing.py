import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    base_path = "/home/kazuya/hdd/Mitosis_detection/weights"
    # for celltype in ["Fluo-N2DL-HeLa", "bcellnormal"]:
    for celltype in ["bcellnormal"]:
        super_f1s = []
        super_stds = []

        sup_path = f"{base_path}/{celltype}/f1score.npy"
        f1, std = np.load(sup_path)
        super_f1s.append(f1)
        super_stds.append(std)

        for missing_rate in [
            "0.05",
            "0.10",
            "0.15",
            "0.20",
            "0.25",
            "0.30",
            "0.40",
            "0.50",
            "0.60",
            "0.70",
            "0.80",
            "0.90",
        ]:
            missing_path = f"{base_path}/{celltype}/missing/{missing_rate}/f1score.npy"
            f1, std = np.load(missing_path)
            super_f1s.append(f1)
            super_stds.append(std)
        super_f1s = np.array(super_f1s)
        super_stds = np.array(super_stds)

        cp_f1s = []
        cp_stds = []

        sup_path = f"{base_path}/{celltype}/copy_pastebrendv2randp/f1score.npy"
        f1, std = np.load(sup_path)
        cp_f1s.append(f1)
        cp_stds.append(std)

        for missing_rate in [
            "0.05",
            "0.10",
            "0.15",
            "0.20",
            "0.25",
            "0.30",
            "0.40",
            "0.50",
            "0.60",
            "0.70",
            "0.80",
            "0.90",
        ]:
            missing_path = f"{base_path}/{celltype}/missingcp/{missing_rate}/f1score.npy"
            f1, std = np.load(missing_path)
            cp_f1s.append(f1)
            cp_stds.append(std)
        cp_f1s = np.array(cp_f1s)
        cp_stds = np.array(cp_stds)

        # plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
        plt.rcParams["font.family"] = "Arial"  # font familyの設定
        plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
        plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
        plt.rcParams["xtick.labelsize"] = 12  # 軸だけ変更されます。
        plt.rcParams["ytick.labelsize"] = 12  # 軸だけ変更されます
        plt.rcParams["xtick.direction"] = "in"  # x axis in
        plt.rcParams["ytick.direction"] = "in"  # y axis in
        plt.rcParams["axes.linewidth"] = 1.0  # axis line width
        plt.rcParams["axes.grid"] = True  # make grid
        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
        plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
        plt.rcParams["legend.labelspacing"] = 1.0  # 垂直（縦）方向の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = 1.0  # 凡例の線と文字の距離の長さ
        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        plt.rcParams["legend.borderaxespad"] = 0.0  # 凡例の端とグラフの端を合わせる

        plt.figure(figsize=(4, 2.8))
        x_value = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
        plt.plot(x_value, super_f1s, label="Baseline", color="#FF0000")
        plt.fill_between(x_value, super_f1s - super_stds, super_f1s + super_stds, alpha=0.2, color="#FF0000")

        plt.plot(x_value, cp_f1s, label="Ours", color="#00B0F0")
        plt.fill_between(x_value, cp_f1s - cp_stds, cp_f1s + cp_stds, alpha=0.2, color="#00B0F0")
        # plt.xlabel('Missing rate', size = "large")
        # plt.ylabel('F1 score', size = "large")
        plt.xlabel("Missing rate")
        plt.ylabel("F1 score")
        plt.xticks(
            x_value,
            ["0%", "5%", "10%", "15%", "20%", "25%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"],
            position=(0.15, -0.01),
        )
        plt.yticks([0, 0.25, 0.50, 0.75, 1], position=(-0.03, 0))
        plt.xlim(0, 90)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.legend()
        # plt.savefig(f"{celltype}.pdf")
        plt.savefig(f"test.png")

        print(1)
    # f1_eachcell = {celltype: [] for celltype in ['Fluo-N2DL-HeLa', 'bcelleasy_cell', 'bcelldet_cell', 'bcellnormal']}
    # f1_allcell = []

    # for train_type in ['/missing']:
    #     for missing_rate in ['0.05', '0.10', '0.15', '0.20', '0.25']:
    #         f1_on_this_rate = 0
    #         for celltype in ['Fluo-N2DL-HeLa', 'bcelleasy_cell', 'bcelldet_cell', 'bcellnormal']:
    #             metric_path = f'{base_path}/{celltype}{train_type}{missing_rate}/super_f1score.npy'
    #             f1 = np.load(metric_path)
    #             f1_eachcell[celltype].append(f1)

    #             f1_on_this_rate += f1
    #         f1_on_this_rate /= 4
    #         f1_allcell.append(f1_on_this_rate)

    # plt.plot([0.05, 0.1, 0.15, 0.2, 0.25], f1_allcell)
    # plt.savefig('test.png')
