import numpy as np
from PIL import Image
import os
total_files = [x for x in os.listdir("/home/jigao/CLIP_witch/poisoning-gradient-matching/poisons") if ".npy" in x]
for outer_idx, j in enumerate(total_files):
    a = np.load("/home/jigao/CLIP_witch/poisoning-gradient-matching/poisons/" + j)
    for idx, i in enumerate(a):
        PIL_image = Image.fromarray(np.uint8(i)).convert('RGB')
        PIL_image.save("/home/hyang/clip_witcher/CLIP/poisoned_data/100K_exp/{}-{}.png".format(outer_idx, idx))