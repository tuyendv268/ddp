from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn

from dataset import (
    Infer_Pairwise_Dataset
)

AUTH_TOKEN = "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"

class Cross_Model(nn.Module):
    def __init__(self, model, tokenizer, max_length=384, droprate=0.2, batch_size=16, device="cpu"):
        super(Cross_Model, self).__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        
        self.model = model
        self.tokenizer = tokenizer
        
        self.dropout = nn.Dropout(droprate)
        self.fc = nn.Linear(768, 1).to(self.device)
        self.cre = torch.nn.CrossEntropyLoss()

    def forward(self, ids, masks, labels=None):
        out = self.model(input_ids=ids,
                         attention_mask=masks)
        out = out.last_hidden_state[:, 0]
        embedding = self.dropout(out)
        logits = self.fc(embedding)
        if labels is not None:
            logits = logits.reshape(labels.size(0), labels.size(1))
            return logits, self.loss(labels=labels, logits=logits)
        
        return logits
    
    def loss(self, labels, logits):
        loss = self.cre(logits, labels.float())
        
        return loss
    
    @torch.no_grad()
    def ranking(self, question, texts):
        tmp = pd.DataFrame()
        tmp["text"] = [" ".join(x.split()) for x in texts]
        tmp["question"] = question
        valid_dataset = Infer_Pairwise_Dataset(
            tmp, self.tokenizer, self.max_length)
        
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=valid_dataset.infer_collate_fn,
            num_workers=0, shuffle=False, pin_memory=True)
        preds = []
        with torch.no_grad():
            bar = enumerate(valid_loader)
            for step, data in bar:
                ids = data["ids"].to(self.device)
                masks = data["masks"].to(self.device)
                preds.append(torch.sigmoid(self(ids, masks)).view(-1))
            preds = torch.concat(preds)
            
        return preds.cpu().numpy()