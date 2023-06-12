from tqdm.auto import tqdm
tqdm.pandas()
import re
from torch.optim import Adam 
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
from pandarallel import pandarallel
from glob import glob
import json

def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    
    return set(lines)

path = "vietnamese-stopword.txt"
stopwords = load_stopwords(path)

def check(sample):
    count=0
    if len(sample["passages"]) > 8:
        return False
    for passage in sample["passages"]:
        if passage["is_selected"] == 1:
            count+=1

    if count == 0 or count > 1:
        return False
    return True

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = []
        for line in f.readlines():
            json_obj = json.loads(line.strip())
            if check(json_obj):
                data.append(line)
    return data

def save_data(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for sample in data:
            json_obj = json.dumps(sample, ensure_ascii=False)
            f.write(json_obj+"\n")

def load_data(path):
    data = []
    for _file in glob(path+"/*.json"):
        data += load_file(_file)
      
    return tuple(data)

def optimizer_scheduler(model, num_train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    optimizer = Adam(optimizer_parameters, lr=5e-6, betas=(0.9, 0.999))
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_train_steps)
    return optimizer, scheduler

def norm_text(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ_0-9\s]+', 
        ' ', text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    return text


def norm_question(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ0-9\;\-\,\.\?\!\/\\\:\(\)+\s]+', 
        ' ', text)
    text = re.sub("\n+", "\n", text)
    text = re.sub("\s+", " ", text)
    text = re.sub("(?<=[a-zA-Z])\.(?=[a-zA-Z])", " ", text)
    text = re.sub("(?<=[a-zA-Z])\,(?=[a-zA-Z])", " ", text)
    return text


def norm_negative_sample(negative_samples):
    normed_negative_samples = []
    for sample in negative_samples:
        normed_negative_samples.append(norm_text(sample))
    return normed_negative_samples