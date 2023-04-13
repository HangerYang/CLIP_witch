import os

# models = ['100K_one_poison_1', 'clean_only', 'clean_only_2', 'poison_only_1',
#             'clean_100k', 'clean_only_1', 'poison_100K', 'poison_only_2']
models = ['100K_cat', '100K_truck']
targets = ['selected_subset_cat_1000_target_ids', 'selected_subset_truck_1000_target_ids']
model_args = ['--model_name %s' % model for model in models]

for i in range(len(model_args)):
    _ = os.system('python verify_text_with_targets.py --device cuda:7 --run_name target \
        --eval_test_data_dir ../poisoning-gradient-matching/downloaded_data/cifar-10-batches-py \
        --targets_path ../poisoning-gradient-matching/poisons/%s.log \
        --model_dir ../../../hyang/clip_witcher/CLIP/logs \
        --end_epoch 32 ' % targets[i]
         + model_args[i])

for args in model_args:
    _ = os.system('python verify_text_with_csv.py --device 7 --run_name all \
         --path ../poisoning-gradient-matching/data_csv/clean_subset_99000.csv \
         --model_dir ../../../hyang/clip_witcher/CLIP/logs \
         --end_epoch 32 '
         + args)