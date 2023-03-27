from pkgs.openai.clip import load as load_model
from PIL import Image
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from pathlib import Path
from tqdm import tqdm
from src.data import ImageCaptionDatasetOrig
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--run_name", type = str, default = "default")
parser.add_argument("--start_epoch", type = int, default = 1)
parser.add_argument("--end_epoch", type = int, default = 64)
parser.add_argument("--batch_size", type = int, default = 1024)
parser.add_argument("--device", type = str, default = "7")
parser.add_argument("--path", type = str, default = "plane")

options = parser.parse_args()

model_name=options.model_name
start = options.start_epoch
end = options.end_epoch
path= options.path
device = 'cuda:{}'.format(options.device)
batch_size = options.batch_size

delimiter=','
image_key="path"
caption_key="caption"


k_total = []
for epoch in tqdm(range(start, end)):
    epoch = str(epoch)
    model, processor = load_model(name = 'RN50', pretrained = False)

    dataset = ImageCaptionDatasetOrig(path, image_key=image_key, caption_key=caption_key, delimiter=delimiter, processor=processor, inmodal=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=16)

    pretrained_path = "logs/{}/checkpoints/epoch_{}.pt".format(model_name, epoch)
    checkpoint = torch.load(pretrained_path, map_location = device)
    
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    df = pd.read_csv(path, sep = delimiter)
    images = df[image_key].tolist()
    acc = []

    model.eval().to(device)
    with torch.no_grad():
        for index, batch in enumerate(dataloader): 
            input_ids, attention_mask, pixel_values = \
                        batch["input_ids"].to(device, non_blocking = True), \
                        batch["attention_mask"].to(device, non_blocking = True), \
                        batch["pixel_values"].to(device, non_blocking = True)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, pixel_values = pixel_values)
            a = outputs[0].cpu().numpy()
            b = outputs[1].cpu().numpy()
            
            k=np.diagonal(cosine_similarity(a, b))
            acc.append(k)
    k_total.append(np.concatenate(acc))



np.save("save_verify_text_with_csv/{}_{}.npy".format(options.model_name, options.run_name), np.array(k_total))
