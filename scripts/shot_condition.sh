#!/bin/bash

############### Copy and paste traininig on 1~7 shots
# copy and paste training
for seed in 0 1 2 3 4
do
    for shot in 1 3 5 7
    do
        for cv_num in 0 1 2 3
        do     
            cell_type=Fluo-N2DL-HeLa
            # Two input two output model train on cell tracking challenge data
            CUDA_VISIBLE_DEVICES=1 python ./scripts/cross_train.py --img_dir ./datas/ctc_preprocessed/${cell_type}\
                --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
                --save_path ./outputs/${cell_type}/newcopy_pastev2/shot${shot}/seed${seed}/${cv_num}\
                --weight_path ./weights/${cell_type}/newcopy_pastev2/shot${shot}/seed${seed}/${cv_num}/best.pth \
                --dataloader CVCPLoaderBrend2

            CUDA_VISIBLE_DEVICES=0 python ./scripts/cross_train.py --img_dir ./datas/bcell_preprocessed/ES\
                --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
                --save_path ./outputs/copy_paste/ES/copy_paste/shot${shot}/seed${seed}/${cv_num}\
                --weight_path ./weights/copy_paste/ES/copy_paste/shot${shot}/seed${seed}/${cv_num}/best.pth\
                --dataloader CVCPLoaderBrend2 &

            CUDA_VISIBLE_DEVICES=1 python ./scripts/cross_train.py --img_dir ./datas/bcell_preprocessed/Fib\
                --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
                --save_path ./outputs/Fib/copy_paste/shot${shot}/seed${seed}/${cv_num}\
                --weight_path ./weights/Fib/copy_paste/shot${shot}/seed${seed}/${cv_num}/best.pth\
                --dataloader CVCPLoaderBrend2 &

            CUDA_VISIBLE_DEVICES=2 python ./scripts/cross_train.py --img_dir ./datas/bcell_preprocessed/ES=D\
                --cv_num ${cv_num} --seed ${seed} --shot ${shot}\
                --save_path ./outputs/ES-D/copy_paste/shot${shot}/seed${seed}/${cv_num}\
                --weight_path ./weights/ES-D/copy_paste/shot${shot}/seed${seed}/${cv_num}/best.pth\
                --dataloader CVCPLoaderBrend2
                wait
        done
    done
done

