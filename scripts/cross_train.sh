#!/bin/bash

# 5 shot comparison
for cv_num in 0 1 2 3
do 
for seed in 0 1 2 3 4
    do
    shot=5
    paste=5
    method=ours
    cell_type=Fluo-N2DL-HeLa
    CUDA_VISIBLE_DEVICES=2 python ./scripts/cross_train.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
        --save_path ./outputs/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader FrameOrderFlippingAlphaBrendingPastingLoader


    cell_type=ES
    CUDA_VISIBLE_DEVICES=2 python ./scripts/cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
        --save_path ./outputs/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader FrameOrderFlippingAlphaBrendingPastingLoader &
    
    cell_type=Fib
    CUDA_VISIBLE_DEVICES=0 python cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
        --save_path ./outputs/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader FrameOrderFlippingAlphaBrendingPastingLoader 
    wait

    cell_type=ES-D
    CUDA_VISIBLE_DEVICES=2 python cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
        --save_path ./outputs/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader FrameOrderFlippingAlphaBrendingPastingLoader
    wait
done 
done

python ./scripts/summalize_result.py --method ours
        
# supervised augmentation
for cv_num in 0 1 2 3
do 
for seed in 0 1 2 3 4
    do
    shot=5
    paste=5
    method=supervised
    cell_type=Fluo-N2DL-HeLa
    CUDA_VISIBLE_DEVICES=2 python ./scripts/cross_train.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
        --save_path ./outputs/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader SupervisedOurloader


    cell_type=normal
    CUDA_VISIBLE_DEVICES=2 python cross_train.py --img_dir ./datas/bcell_preprocessed/normal_type\
        --cv_num ${cv_num} --seed ${seed} \
        --save_path ./outputs/bcell${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader SupervisedOurloader &
    
    cell_type=easy_cell
    CUDA_VISIBLE_DEVICES=0 python cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} \
        --save_path ./outputs/bcell${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader SupervisedOurloader 
    wait

    cell_type=det_cell
    CUDA_VISIBLE_DEVICES=2 python cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} \
        --save_path ./outputs/bcell${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/${method}/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader SupervisedOurloader
    wait
done 
done


################ fully Supervised traininig
for cv_num in 0 1 2 3
do 
    cell_type=Fluo-N2DL-HeLa
    # Two input two output model train on cell tracking challenge data
    CUDA_VISIBLE_DEVICES=0 python cross_train.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot} \
        --save_path ./outputs/${cell_type}/supervised/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/${cell_type}/supervised/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader CVTrainloader &

    CUDA_VISIBLE_DEVICES=1 python cross_train.py --img_dir ./datas/bcell_preprocessed/normal_type\
        --cv_num ${cv_num} --seed ${seed} --shot ${shot} \
        --save_path ./outputs/bcellnormal/copy_paste/shot${shot}/seed${seed}/${cv_num}\
        --weight_path ./weights/bcellnormal/copy_paste/shot${shot}/seed${seed}/${cv_num}/best.pth\
        --dataloader CVTrainloader &
    
    cell_type=easy_cell
    CUDA_VISIBLE_DEVICES=0 python cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
        --cv_num ${cv_num}\
        --save_path ./outputs/bcell${cell_type}/balanced/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/balanced/${cv_num}/best.pth\
        --dataloader CVbalancedTrainloader &
    
    cell_type=det_cell
    CUDA_VISIBLE_DEVICES=1 python cross_train.py --img_dir ./datas/bcell_preprocessed/${cell_type} \
        --cv_num ${cv_num}\
        --save_path ./outputs/bcell${cell_type}/balanced/${cv_num}\
        --weight_path ./weights/bcell${cell_type}/balanced/${cv_num}/best.pth\
        --dataloader CVbalancedTrainloader
done