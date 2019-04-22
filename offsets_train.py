import time
import os
import random
import argparse
import json

import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import log_loss

from utility import get_gap_model, get_offsets_loader, get_param_size, run_epoch


t0 = time.time()

parser = argparse.ArgumentParser()

parser.add_argument("--model_id",
                    default=999, type=int, required=False)

parser.add_argument("--bert_model", default="bert-large-cased", type=str, required=False,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, ")
parser.add_argument("--task_name",
                    default="GAP: Offsets model",
                    type=str,
                    required=False,
                    help="The name of the task to train.")

# Preprocessing
parser.add_argument("--do_lower_case",
                    default=False,
                    action="store_true")

# Model
parser.add_argument("--n_bertlayers",
                    default=22, type=int, required=False)
parser.add_argument("--dropout",
                    type=float,
                    default=0.1)

# Training
parser.add_argument("--train_batch_size",
                    default=1,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--gradient_accumulation_steps",
                    type=int,
                    default=20,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")
parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--lr",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=2,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                    "E.g., 0.1 = 10%% of training.")
parser.add_argument("--optim",
                    default="bertadam",
                    type=str,
                    help="Optimizer: bertadam or adam")
parser.add_argument("--weight_decay",
                    default=False,
                    action="store_true")

parser.add_argument("--save_model",
                    default=False,
                    action="store_true")

parser.add_argument("--model_fname_template",
                    default="offsets_{:03d}_ep_{:d}.model",
                    type=str)

# DIR
parser.add_argument("--log_dir",
                    default="log/", type=str, required=False)
parser.add_argument("--data_dir",
                    default="./", type=str, required=False)
parser.add_argument("--model_dir",
                    default="model/",
                    type=str,
                    required=False)

args = parser.parse_args()

args.n_models = args.model_id + 1
model_id = args.model_id
del args.model_id

with open("offsets_cfg.json", "w") as f:
    json.dump(vars(args), f, indent=4)

print(args)

print("\nEffective batch_size:", args.train_batch_size * args.gradient_accumulation_steps)

device = torch.device("cuda")


seed = model_id + 1000

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

data_dir = args.data_dir


def eval_model(model, dataloader, y):
    pr = model.predict(dataloader)
    loss = log_loss(y, pr)
    return loss


all_df = pd.concat([pd.read_csv(data_dir + "gap-test.tsv", index_col=0, delimiter="\t"),
                    pd.read_csv(data_dir + "gap-validation.tsv", index_col=0, delimiter="\t"),
                    pd.read_csv(data_dir + "gap-development.tsv", index_col=0, delimiter="\t")])
all_df = all_df.reset_index(drop=True)


y_df = all_df[["A-coref", "B-coref"]].astype(int)
y_df["Neither"] = 1 - y_df.sum(1)
all_df["y"] = y = np.argmax(y_df[["A-coref", "B-coref", "Neither"]].values, 1)

epochs = args.num_train_epochs
gradient_accumulation_steps = args.gradient_accumulation_steps

tr_loader, tr_eval_loader, _ = get_offsets_loader(args, train_df=all_df)

steps_per_epoch = len(tr_loader)
print("steps_per_epoch:", steps_per_epoch)

model, optimizer = get_gap_model(args, steps_per_epoch, device)
total_psize, trainalbe_psize = get_param_size(model)
print(f"Total params: {total_psize}\nTrainable params: {trainalbe_psize}")

criterion = CrossEntropyLoss()
scores = []
for j in range(epochs):
    epoch = j + 1
    t1 = time.time()
    model.train()
    cum_tr_loss = run_epoch(model, tr_loader, optimizer, criterion,
                            gradient_accumulation_steps, device=device)

    model.eval()
    tr_loss = eval_model(model, tr_eval_loader, y)
    elapsed = time.time() - t1
    sc = {"epoch": epoch, "time": elapsed,
          "cum_tr_loss": cum_tr_loss, "tr_loss": tr_loss}
    scores.append(sc)
    print(f"Epoch:{epoch} cum_tr_loss:{cum_tr_loss:.4f} tr_loss:{tr_loss:.4f}"
          f" time:{elapsed:.0f}s")
    if args.save_model:
        fname = args.model_dir + args.model_fname_template.format(model_id, epoch)
        torch.save(model.state_dict(), fname)
        print("\nSaved:", fname)

df = pd.DataFrame(scores)
df.to_csv(args.log_dir + f"offsets_{model_id:03d}.csv")

print(f"Done model_id:{model_id:03d} {(time.time() - t0) / 60:.1f} minutes\n\n")
