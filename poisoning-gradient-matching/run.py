import os

list_scripts = ['brew_poison.py  --pretrained --paugment --budget=1 --restarts=1 --save=numpy --attackiter 5  '
                '--targetclass %d --train_data data_csv/poison_subset%d.csv' % (i,i) for i in range(10)]

for script in list_scripts:
    __ = os.system('python ' + script)
