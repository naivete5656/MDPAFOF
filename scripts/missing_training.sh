#!/bin/bash

################ Supervised
for cv_num in 0 1 2 3
do 
    for seed in 0 1 2 3 4
    do 
        for missing in 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 
        do
        cell_type=Fluo-N2DL-HeLa
        # Two input two output model train on cell tracking challenge data
        CUDA_VISIBLE_DEVICES=1 python missing_training.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/${cell_type}/missing/${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/${cell_type}/missing/${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingSuperLoader &

        cell_type=normal
        CUDA_VISIBLE_DEVICES=0 python ./scripts/missing_training.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/bcell${cell_type}/missing/${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/bcell${cell_type}/missing/${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingSuperLoader
        wait

        cell_type=easy_cell
        CUDA_VISIBLE_DEVICES=1 python missing_training.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/bcell${cell_type}/missing/${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/bcell${cell_type}/missing/${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingSuperLoader &

        cell_type=det_cell
        CUDA_VISIBLE_DEVICES=2 python missing_training.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/bcell${cell_type}/missing/${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/bcell${cell_type}/missing/${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingSuperLoader
        wait
        done
    done
done


################ copy paste
for cv_num in 0 1 2 3
do 
    for seed in 0 1 2 3 4
    do 
        for missing in 0.40 0.50 0.60 0.70 0.80 0.90 1.00
        do
        cell_type=Fluo-N2DL-HeLa
        # Two input two output model train on cell tracking challenge data
        CUDA_VISIBLE_DEVICES=1 python missing_training.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/${cell_type}/missingcp/${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/${cell_type}/missingcp/${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingCP &

        cell_type=normal
        CUDA_VISIBLE_DEVICES=2 python ./scripts/missing_training.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/bcell${cell_type}/missingcp/${missing}/seed${seed}/${cv_num} \
            --weight_path ./weights/bcell${cell_type}/missingcp/${missing}/seed${seed}/${cv_num}/best.pth \
            --dataloader MissingCP
        wait

        cell_type=easy_cell
        CUDA_VISIBLE_DEVICES=1 python missing_training.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/bcell${cell_type}/missingcp${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/bcell${cell_type}/missingcp${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingCP &

        cell_type=det_cell
        CUDA_VISIBLE_DEVICES=2 python missing_training.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
            --cv_num ${cv_num} --seed ${seed} --missing ${missing} \
            --save_path ./outputs/bcell${cell_type}/missingcp${missing}/seed${seed}/${cv_num}\
            --weight_path ./weights/bcell${cell_type}/missingcp${missing}/seed${seed}/${cv_num}/best.pth\
            --dataloader MissingCP
        wait
        done
    done
done