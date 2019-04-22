import time
import argparse
import json

import numpy as np
import pandas as pd

import torch

from utility import BertClOffsets, get_offsets_loader, objectview

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


with open("offsets_cfg.json") as f:
    cfg = objectview(json.load(f))

_, _, test_loader = get_offsets_loader(cfg, test_df=test_df)

model = BertClOffsets(bert_model=cfg.bert_model,
                      n_bertlayers=cfg.n_bertlayers,
                      dropout=cfg.dropout)
model.to(device)
model.eval()


n_models = cfg.n_models
test_pr = np.zeros((len(test_df), 3))

print("\nPredicting..")
for model_id in range(n_models):
    print(f"model_id: {model_id}")
    for epoch in [1, 2]:
        t1 = time.time()
        fname = cfg.model_dir + cfg.model_fname_template.format(model_id, epoch)
        model.load_state_dict(torch.load(fname))
        test_pr += model.predict(test_loader)
        print(f"  epoch: {epoch} time: {time.time() - t1:.1f}s")

test_pr /= 2 * n_models

df_sub = pd.DataFrame(test_pr, columns=["A", "B", "NEITHER"])
df_sub["ID"] = test_df.ID
df_sub = df_sub[["ID", "A", "B", "NEITHER"]]

df_sub.to_csv("offsets_v1.csv", index=False)

print(f"\n{(time.time() - t0) / 60:.2f} minutes\n\n")
