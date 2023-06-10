from tqdm.auto import tqdm
tqdm.pandas()
import re
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from pandarallel import pandarallel
from glob import glob
import json

def check(sample):
    count=0 
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
                data.append(json_obj)
            
    return data

def load_data(path):
    data = []
    for _file in glob(path+"/*.json"):
        data += load_file(_file)
            
    return data

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

    opt = AdamW(optimizer_parameters, lr=4e-5, no_deprecation_warning=True)
    sch = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.05*num_train_steps),
        num_training_steps=num_train_steps,
        last_epoch=-1,
    )
    return opt, sch

def norm_text(text):
    text = text.lower()
    text = re.sub(
        r'[^a-zaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ0-9\s]+', 
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