SPLITS=5

for METHOD in 'linear_head'
do
    for MODEL in 'candid_ptx_spt_gloria_pretrain' 'candid_ptx_dapt_gloria_pretrain'
    do
        for TRAIN_PCT in 0.01 0.05 0.1 0.2
        do
            python run.py -c "./configs/candid_ptx_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct $TRAIN_PCT --splits $SPLITS
        done
        python run.py -c "./configs/candid_ptx_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct 1
    done
done

# for METHOD in 'linear_head' 'end_to_end'
# do
#     for MODEL in 'imagenet_supervised_pretrain' 'chexpert_supervised_pretrain' 'chexpert_moco_pretrain' 'chexpert_gloria_pretrain' 'candid_ptx_spt_gloria_pretrain' 'candid_ptx_dapt_gloria_pretrain'
#     do
#         for TRAIN_PCT in 0.01 0.05 0.1 0.2
#         do
#             python run.py -c "./configs/candid_ptx_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct $TRAIN_PCT --splits $SPLITS
#         done
#         python run.py -c "./configs/candid_ptx_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct 1
#     done
# done