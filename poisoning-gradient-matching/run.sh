#!/bin/bash



poisonkeys=(1000000 1200000 1230000 1231000 1231200 1231230 1231231)

target_class=4 
intended_class=9
device=5
train_data='data_csv/classification/selected_subset_truck_1000.csv'

for poisonkey in "${poisonkeys[@]}"
do
    python brew_poison.py  --pretrained --paugment --budget=1 --restarts=4 --save=csv --attackiter 250 --targetclass $target_class --intendedclass $intended_class --targets 1 --device $device --root data_csv --train_data $train_data --poisonkey $poisonkey
    wait
done