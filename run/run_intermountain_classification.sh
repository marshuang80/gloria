
SPLITS=5

METHOD='linear_head'

python run.py -c "./configs/intermountain_classification/${METHOD}/chexpert_supervised_pretrain_config.yaml" --train --test --train_pct 0.1 --start_split 4 --splits $SPLITS
python run.py -c "./configs/intermountain_classification/${METHOD}/chexpert_gloria_pretrain_config.yaml" --train --test --train_pct 0.1 --start_split 1 --splits $SPLITS
python run.py -c "./configs/intermountain_classification/${METHOD}/chexpert_gloria_pretrain_config.yaml" --train --test --train_pct 0.2 --start_split 4 --splits $SPLITS

# for METHOD in 'linear_head'
# do
#     for MODEL in 'intermountain_spt_gloria_pretrain' 'intermountain_dapt_gloria_pretrain'
#     do
#         for TRAIN_PCT in 0.01 0.05 0.1 0.2
#         do
#             python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct $TRAIN_PCT --splits $SPLITS
#         done
#         python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct 1
#     done
# done

# for METHOD in 'linear_head' 'end_to_end'
# do
#     for MODEL in 'imagenet_supervised_pretrain' 'chexpert_supervised_pretrain' 'chexpert_moco_pretrain' 'chexpert_gloria_pretrain' 'intermountain_spt_gloria_pretrain' 'intermountain_dapt_gloria_pretrain'
#     do
#         for TRAIN_PCT in 0.01 0.05 0.1 0.2
#         do
#             python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct $TRAIN_PCT --splits $SPLITS
#         done
#         python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct 1
#     done
# done