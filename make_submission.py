import pandas as pd


offsets_pr = pd.read_csv("offsets_v1.csv")
nli_pr = pd.read_csv("nli_v1.csv")

keys = offsets_pr.columns[1:]
pr = nli_pr.copy()
pr[keys] = 0.7 * offsets_pr[keys] + 0.3 * nli_pr[keys]


offsets_pr.to_csv("subm001.csv", index=False)
pr.to_csv("subm002.csv", index=False)
