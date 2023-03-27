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
from src.data import get_eval_target_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "default")
parser.add_argument("--run_name", type = str, default = "default")
parser.add_argument("--start_epoch", type = int, default = 1)
parser.add_argument("--end_epoch", type = int, default = 64)
parser.add_argument("--batch_size", type = int, default = 1024)
parser.add_argument("--num_workers", type = int, default = 16)
parser.add_argument("--device", type = str, default = "cuda:7")
parser.add_argument("--targets_path", type = str, default = None)
parser.add_argument("--eval_data_type", type = str, default = 'CIFAR10')
parser.add_argument("--eval_test_data_dir", type = str, default = None)

options = parser.parse_args()

model_name=options.model_name
start = options.start_epoch
end = options.end_epoch

f = open(options.targets_path, "r")
lines = f.readlines()
targets = {}

for line in lines:
    if line != '':
        t = line.split(',')
        targets[int(t[0])] = int(t[1])


k_total = []
for epoch in tqdm(range(start, end)):
    epoch = str(epoch)
    
    model, processor = load_model(name = 'RN50', pretrained = False)

    pretrained_path = "logs/{}/checkpoints/epoch_{}.pt".format(model_name, epoch)
    checkpoint = torch.load(pretrained_path, map_location = options.device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    dataloader = get_eval_target_dataloader(options, processor, targets)
    acc = []

    model.eval().to(options.device)
    umodel = model

    config = eval(open(f"{options.eval_test_data_dir}/classes.py", "r").read())
    classes, templates = config["classes"], config["templates"]
    with torch.no_grad():
        text_embeddings = []
        for c in classes:
            text = [template(c) for template in templates]
            text_tokens = processor.process_text(text)
            text_input_ids, text_attention_mask = text_tokens["input_ids"].to(options.device), text_tokens["attention_mask"].to(options.device) 
            text_embedding = umodel.get_text_features(input_ids = text_input_ids, attention_mask = text_attention_mask)
            text_embedding /= text_embedding.norm(dim = -1, keepdim = True)
            text_embedding = text_embedding.mean(dim = 0)
            text_embedding /= text_embedding.norm()
            text_embeddings.append(text_embedding)
        text_embeddings = torch.stack(text_embeddings, dim = 0).cpu().numpy()
    # import pdb
    # pdb.set_trace()
    with torch.no_grad():
        for image, target_label, orig_label in dataloader:
            image, target_label, orig_label = image.to(options.device), target_label.to(options.device), orig_label.to(options.device)
            image_embedding = umodel.get_image_features(image)
            image_embedding /= image_embedding.norm(dim = -1, keepdim = True)
            image_embedding = image_embedding.cpu().numpy()

            target_embeddings = []
            for c in target_label:
                target_embeddings.append(text_embeddings[c])
            target_embeddings = np.array(target_embeddings)
            
            k=np.diagonal(cosine_similarity(image_embedding, target_embeddings))
            acc.append(k)
    k_total.append(np.concatenate(acc))


np.save("save_verify_text_with_csv/{}_{}.npy".format(options.model_name, options.run_name), np.array(k_total))
