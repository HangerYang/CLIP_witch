#!/bin/bash



poisonkeys=(1000000)
# 1200000 1230000 1231000 1231200 1231230 1231231)

target_class=deer
intended_class=truck
device=5
targetn=1
restarts=4
attkiter=250
datatype='classification'

for poisonkey in "${poisonkeys[@]}"
do
    target_intend="target_${target_class}_intend_${intended_class}"
    hyperparam="targetn_${targetn}_restart_${restarts}_attkiter_${attkiter}_${datatype}_${poisonkey}"
    
    poison_data="poisons/${target_intend}/${hyperparam}/poisoned_data.csv"
    clean_data="data_csv/${datatype}/clean_subset_99000.csv"
    train_data="poisons/${target_intend}/${hyperparam}/poisoned_train_data.csv"
    
    if [ -f $train_data ]; then
        echo "$train_data exists. Doing nothing."
    else
        tail -n +2 $clean_data > temp 
        wait
        cat $poison_data temp > $train_data 
        wait
        rm temp
        wait
    fi

    python -m src.main --name "${target_intend}_${hyperparam}" --train_data $train_data --image_key path --caption_key caption --device_id $device  --batch_size 256  --epoch 64 --checkpoint "logs/${target_intend}_${hyperparam}/checkpoints/epoch_32.pt"

    wait
done