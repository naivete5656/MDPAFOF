#!/bin/bash

for seed in 0 1 2 3 4
do
    for cv_num in 0 1 2 3
    do 
    shot=1

    cell_type=Fluo-N2DL-HeLa
    # Two input two output model train on cell tracking challenge data
    CUDA_VISIBLE_DEVICES=0 python ./scripts/cross_test.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed}\
        --save_path ./outputs/forpu/${cell_type}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/partialsup/shot${shot}/seed${seed}/${cv_num}/best.pth

    cell_type=normal
    CUDA_VISIBLE_DEVICES=0 python ./scripts/cross_test.py --img_dir ./datas/bcell_preprocessed/normal \
        --cv_num ${cv_num}\
       --save_path ./outputs/forpu/bcell${cell_type}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/partialsup/shot${shot}/seed${seed}/${cv_num}/best.pth
    
    cell_type=easy_cell
    CUDA_VISIBLE_DEVICES=0 python ./scripts/cross_test.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
        --cv_num ${cv_num}\
       --save_path ./outputs/forpu/bcell${cell_type}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/partialsup/shot${shot}/seed${seed}/${cv_num}/best.pth
    
    cell_type=det_cell
    CUDA_VISIBLE_DEVICES=0 python ./scripts/cross_test.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
        --cv_num ${cv_num}\
       --save_path ./outputs/forpu/bcell${cell_type}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/partialsup/shot${shot}/seed${seed}/${cv_num}/best.pth
    
    done
done