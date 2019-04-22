import time
import argparse
import json

import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import normalize

from utility import BertCl, get_nli_loader, objectview

t0 = time.time()

device = torch.device("cuda")

parser = argparse.ArgumentParser()

parser.add_argument("--test_file",
                    default="",
                    type=str,
                    required=True)

args = parser.parse_args()

test_df = pd.read_csv(args.test_file, delimiter="\t")
print(args.test_file, ":", test_df.shape)


with open("nli_cfg.json") as f:
    cfg = objectview(json.load(f))

print("Preprocessing..")
_, _, test_loader = get_nli_loader(cfg, test_df=test_df)

model = BertCl(bert_model=cfg.bert_model,
               n_bertlayers=cfg.n_bertlayers,
               dropout=cfg.dropout)
model.to(device)
model.eval()

epoch = 2
n_models = cfg.n_models
test_pr = np.zeros((len(test_df), 3))
print("\nPredicting..")
for model_id in range(n_models):
    t1 = time.time()
    df = pd.read_csv(f"log/nli_{model_id:03d}.csv")
    tr_loss = df["tr_loss"][1]
    print(f"model_id: {model_id} tr_loss:{tr_loss:.4f} at epoch {epoch}")
    if tr_loss > 0.2:
        print(f"Skip model_id:{model_id} due to higher training loss.")
        continue
    fname = cfg.model_dir + cfg.model_fname_template.format(model_id, epoch)
    model.load_state_dict(torch.load(fname))
    test_pr += model.predict(test_loader).reshape((-1, 3))
    print(f"  time: {time.time() - t1:.1f}s")

test_pr = normalize(test_pr, norm="l1")

df_sub = pd.DataFrame(test_pr, columns=["A", "B", "NEITHER"])
df_sub["ID"] = test_df.ID
df_sub = df_sub[["ID", "A", "B", "NEITHER"]]

df_sub.to_csv("nli_v1.csv", index=False)

print(f"\n{(time.time() - t0) / 60:.2f} minutes\n\n")
