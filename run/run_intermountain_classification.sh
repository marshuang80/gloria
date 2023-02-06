SPLITS=5

for METHOD in 'linear_head'
do
    for MODEL in 'intermountain_spt_moco_pretrain'
    do
        for TRAIN_PCT in 0.1
        do
            python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct $TRAIN_PCT --splits $SPLITS
        done
        python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct 1
    done
done

for METHOD in 'linear_head'
do
    for MODEL in 'intermountain_dapt_moco_pretrain' 'intermountain_spt_gloria_pretrain' 'intermountain_dapt_gloria_pretrain'
    do
        for TRAIN_PCT in 0.01 0.1
        do
            python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct $TRAIN_PCT --splits $SPLITS
        done
        python run.py -c "./configs/intermountain_classification/${METHOD}/${MODEL}_config.yaml" --train --test --train_pct 1
    done
done