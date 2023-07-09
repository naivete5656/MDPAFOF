from pathlib import Path
from statistics import mean, stdev
import numpy as np


def supervised_training():
    base_path = "/home/kazuya/hdd/Mitosis_detection/weights"
    # for celltype in ['Fluo-N2DL-HeLa','bcelleasy_cell','bcelldet_cell', 'bcellnormal']:
    for celltype in ["Fluo-N2DL-HeLa", "bcellnormal"]:
        # for celltype in ['bcelldet_cell']:

        # for train_type in ['/supervised_augmentation']:

        f1_ave, pre_ave, rec_ave = [], [], []
        for cv_num in range(4):
            # log_path = Path(f'{base_path}/{celltype}/balanced/{cv_num}/training_log.log')
            log_path = Path(f"{base_path}/{celltype}/{cv_num}/training_log.log")

            with log_path.open("r") as f:
                alltxt = f.readlines()
                result = alltxt[-1]
            try:
                f1, precision, recall = result.split(",")
                f1, precision, recall = float(f1), float(precision), float(recall.replace("\n", ""))
            except:
                0
            f1_ave.append(f1)
            pre_ave.append(precision)
            rec_ave.append(recall)
        f1_std = stdev(f1_ave)
        f1_ave = mean(f1_ave)
        pre_ave = mean(pre_ave)
        rec_ave = mean(rec_ave)

        save_path = f"{base_path}/{celltype}/f1score"
        np.save(save_path, [f1_ave, f1_std])
        print(f"celltype, f1, pre, rec")
        print(f"{celltype}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}")


# def copy_paste_average():
#     base_path = '/home/kazuya/hdd/Mitosis_detection/weights'

#     for celltype in ['Fluo-N2DL-HeLa', 'bcellnormal']:
#     # for celltype in ['Fluo-N2DL-HeLa', 'bcelleasy_cell', 'bcelldet_cell', 'bcellnormal']:


#         # /home/kazuya/hdd/Mitosis_detection/weights/Fluo-N2DL-HeLa/copy_paste
#     # for celltype in ['Fluo-N2DL-HeLa']:
#         for train_type in ['/copy_pastebrendv2randp']:
#         # for train_type in ['/contrastive_copypa_triplet']:
#         # for train_type in ['/newcopy_paste/shot1', '/newcopy_paste/shot3', '/newcopy_paste/shot5', '/newcopy_paste/shot7']:
#             for shot in [5]:
#                 f1_ave, pre_ave, rec_ave = [], [], []
#                 for seed in range(5):
#                     for cv_num in range(4):
#                         # log_path = Path(f'{base_path}/{celltype}{train_type}/shotfull/seed{seed}/{cv_num}/training_log.log')
#                         log_path = Path(f'{base_path}/{celltype}{train_type}/shot{shot}/seed{seed}/{cv_num}/training_log.log')
#                         # log_path = Path(f'{base_path}/{celltype}{train_type}/seed{seed}/{cv_num}/training_log.log')
#                         with log_path.open('r') as f:
#                             alltxt = f.readlines()
#                             result = alltxt[-1]
#                         try:
#                             f1, precision, recall = result.split(',')
#                             f1, precision, recall = float(f1), float(precision), float(recall.replace('\n', ''))
#                         except:
#                             f1, precision, recall = 0, 0, 0

#                         f1_ave.append(f1)
#                         pre_ave.append(precision)
#                         rec_ave.append(recall)
#                 f1_std = stdev(f1_ave)
#                 f1_ave = mean(f1_ave)
#                 pre_ave = mean(pre_ave)
#                 rec_ave = mean(rec_ave)
#                 save_path = f'{base_path}/{celltype}{train_type}/f1score'
#                 np.save(save_path, [f1_ave, f1_std])
#                 print(f'copy and paste celltype, f1, pre, rec')
#                 print(f'{train_type}, {celltype}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}')


def copy_paste_full_average():
    base_path = "/home/kazuya/hdd/Mitosis_detection/weights"

    for celltype in ["Fluo-N2DL-HeLa", "bcelleasy_cell", "bcelldet_cell", "bcellnormal"]:
        for train_type in ["/copy_paste"]:
            shot = "full"
            f1_ave, pre_ave, rec_ave = [], [], []
            for seed in range(5):
                for cv_num in range(4):
                    # log_path = Path(f'{base_path}/{celltype}{train_type}/shotfull/seed{seed}/{cv_num}/training_log.log')
                    log_path = Path(
                        f"{base_path}/{celltype}{train_type}/shot{shot}/seed{seed}/{cv_num}/training_log.log"
                    )
                    # log_path = Path(f'{base_path}/{celltype}{train_type}/seed{seed}/{cv_num}/training_log.log')
                    with log_path.open("r") as f:
                        alltxt = f.readlines()
                        result = alltxt[-1]
                    try:
                        f1, precision, recall = result.split(",")
                        f1, precision, recall = float(f1), float(precision), float(recall.replace("\n", ""))
                    except:
                        f1, precision, recall = 0, 0, 0

                    f1_ave.append(f1)
                    pre_ave.append(precision)
                    rec_ave.append(recall)
            f1_std = stdev(f1_ave)
            f1_ave = mean(f1_ave)
            pre_ave = mean(pre_ave)
            rec_ave = mean(rec_ave)
            save_path = f"{base_path}/{celltype}{train_type}/f1score"
            np.save(save_path, [f1_ave, f1_std])
            print(f"copy and paste celltype, f1, pre, rec")
            print(f"{train_type}, {celltype}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}")


def shot_average():
    base_path = "/home/kazuya/hdd/Mitosis_detection/weights"

    # for celltype in ['bcelldet_cell', 'bcelleasy_cell']:
    for celltype in ["copy_paste/bcellnormal"]:
        # for train_type in ['/fujiipartial']:
        for train_type in ["/copy_paste"]:
            for shot in [1]:
                f1_ave, pre_ave, rec_ave = [], [], []
                for seed in range(5):
                    for cv_num in range(4):
                        log_path = Path(
                            f"{base_path}/{celltype}{train_type}/shot{shot}/seed{seed}/{cv_num}/training_log.log"
                        )

                        with log_path.open("r") as f:
                            alltxt = f.readlines()
                            result = alltxt[-1]
                        try:
                            f1, precision, recall = result.split(",")
                            f1, precision, recall = float(f1), float(precision), float(recall.replace("\n", ""))
                        except:
                            f1, precision, recall = 0, 0, 0

                        f1_ave.append(f1)
                        pre_ave.append(precision)
                        rec_ave.append(recall)
                f1_ave = mean(f1_ave)
                pre_ave = mean(pre_ave)
                rec_ave = mean(rec_ave)
                print(f"copy and paste celltype, f1, pre, rec")
                print(f"{celltype}, {shot}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}")


def missing_ave():
    base_path = "/home/kazuya/hdd/Mitosis_detection/weights"

    # for celltype in ['Fluo-N2DL-HeLa', 'bcelleasy_cell', 'bcelldet_cell', 'bcellnormal']:
    # for celltype in ['Fluo-N2DL-HeLa', 'bcellnormal']:
    for celltype in ["bcellnormal"]:
        for train_type in ["/missing/", "/missingcp/"]:
            # for missing_rate in ["0.05", "0.10", "0.15", "0.20", "0.25", "0.30"]:
            for missing_rate in ["0.10", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90"]:
                f1_ave, pre_ave, rec_ave = [], [], []
                for seed in range(5):
                    for cv_num in range(4):
                        log_path = Path(
                            f"{base_path}/{celltype}{train_type}{missing_rate}/seed{seed}/{cv_num}/training_log.log"
                        )
                        with log_path.open("r") as f:
                            alltxt = f.readlines()
                            result = alltxt[-1]
                        try:
                            f1, precision, recall = result.split(",")
                            f1, precision, recall = float(f1), float(precision), float(recall.replace("\n", ""))
                        except:
                            f1, precision, recall = 0, 0, 0

                        f1_ave.append(f1)
                        pre_ave.append(precision)
                        rec_ave.append(recall)
                f1_std = stdev(f1_ave)
                f1_ave = mean(f1_ave)
                pre_ave = mean(pre_ave)
                rec_ave = mean(rec_ave)

                save_path = f"{base_path}/{celltype}{train_type}{missing_rate}/f1score"

                np.save(save_path, [f1_ave, f1_std])
                print(f"{train_type} celltype, f1, pre, rec")
                print(f"{missing_rate}, {celltype}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}")


def partial_average():
    base_path = "/home/kazuya/hdd/Mitosis_detection/weights"

    # for celltype in ["Fluo-N2DL-HeLa", "bcelleasy_cell", "bcelldet_cell", "bcellnormal"]:
    for celltype in ["Fluo-N2DL-HeLa", "bcelleasy_cell", "bcelldet_cell", "bcellnormal"]:
        for train_type in ["/partial", "/partialsup"]:
            # for train_type in ['/contrastive_copypa_triplet']:
            # for train_type in ['/newcopy_paste/shot1', '/newcopy_paste/shot3', '/newcopy_paste/shot5', '/newcopy_paste/shot7']:
            for shot in [1]:
                f1_ave, pre_ave, rec_ave = [], [], []
                for seed in range(5):
                    for cv_num in range(4):
                        # log_path = Path(f'{base_path}/{celltype}{train_type}/shotfull/seed{seed}/{cv_num}/training_log.log')
                        log_path = Path(
                            f"{base_path}/{celltype}{train_type}/shot{shot}/seed{seed}/{cv_num}/training_log.log"
                        )
                        # log_path = Path(f'{base_path}/{celltype}{train_type}/seed{seed}/{cv_num}/training_log.log')
                        with log_path.open("r") as f:
                            alltxt = f.readlines()
                            result = alltxt[-1]
                        try:
                            f1, precision, recall = result.split(",")
                            f1, precision, recall = float(f1), float(precision), float(recall.replace("\n", ""))
                        except:
                            f1, precision, recall = 0, 0, 0

                        f1_ave.append(f1)
                        pre_ave.append(precision)
                        rec_ave.append(recall)
                f1_ave = mean(f1_ave)
                pre_ave = mean(pre_ave)
                rec_ave = mean(rec_ave)
                print(f"copy and paste celltype, f1, pre, rec")
                print(f"{train_type}, {celltype}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}")


if __name__ == "__main__":
    # partial_average()
    # supervised_training()
    copy_paste_full_average()
    missing_ave()
    copy_paste_average()
    # shot_average()

    # base_path = '/mnt/d/main/Mitosis_detection/weights'

    # # for celltype in ['bcelldet_cell', 'bcelleasy_cell']:
    # # for celltype in ['Fluo-N2DL-HeLa']:
    # for celltype in ['Fluo-N2DL-HeLa', 'bcellnormal', 'bcelldet_cell', 'bcelleasy_cell']:
    #     print(f'supervised celltype, f1, pre, rec')
    #     # for train_type in ['/super_copy_paste', '/super_copy_paste_sim', '/mean_teacher']:
    #     for train_type in ['/partialsup', '', '/partial']:

    #         f1_ave, pre_ave, rec_ave = [], [], []
    #         f1_ave_super, pre_ave_super, rec_ave_super = [], [], []
    #         for cv_num in range(4):
    #             log_path = Path(f'{base_path}/{celltype}{train_type}/{cv_num}/training_log.log')
    #             with log_path.open('r') as f:
    #                 alltxt = f.readlines()
    #                 result = alltxt[-1]
    #             try:
    #                 f1, precision, recall = result.split(',')
    #                 f1, precision, recall = float(f1), float(precision), float(recall.replace('\n', ''))
    #                 f1_ave_super.append(f1)
    #                 pre_ave_super.append(precision)
    #                 rec_ave_super.append(recall)
    #             except:
    #                 f1_ave_super.append(0)
    #                 pre_ave_super.append(0)
    #                 rec_ave_super.append(0)

    #             # for seed in range(5):
    #             #     log_path = Path(f'{base_path}/{celltype}/copy_paste/shot5/seed{seed}/{cv_num}/training_log.log')

    #             #     with log_path.open('r') as f:
    #             #         alltxt = f.readlines()
    #             #         result = alltxt[-1]
    #             #     f1, precision, recall = result.split(',')
    #             #     f1, precision, recall = float(f1), float(precision), float(recall.replace('\n', ''))
    #             #     f1_ave.append(f1)
    #             #     pre_ave.append(precision)
    #             #     rec_ave.append(recall)
    #         f1_ave_super = mean(f1_ave_super)
    #         pre_ave_super = mean(pre_ave_super)
    #         rec_ave_super = mean(rec_ave_super)
    #         print(f'{train_type}, {celltype}, {f1_ave_super:0.3f}, {pre_ave_super:0.3f}, {rec_ave_super:0.3f}')

    #         # f1_ave = mean(f1_ave)
    #         # pre_ave = mean(pre_ave)
    #         # rec_ave = mean(rec_ave)
    #         # print(f'copy and paste celltype, f1, pre, rec')
    #         # print(f'{train_type}, {celltype}, {f1_ave:0.3f}, {pre_ave:0.3f}, {rec_ave:0.3f}')
