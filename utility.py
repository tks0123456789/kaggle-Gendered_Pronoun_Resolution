import time
import numpy as np
import torch
from torch.nn import Module, Linear, Dropout
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import Adam
from pytorch_pretrained_bert.modeling import BertModel, BertLayer
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
import re
import spacy


device = torch.device("cuda")


def get_param_size(model):
    trainable_psize = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    total_psize = np.sum([np.prod(p.size()) for p in model.parameters()])
    return total_psize, trainable_psize


def insert_tag(row):
    """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""
    to_be_inserted = sorted([
        (row["A-offset"], " [A] "),
        (row["B-offset"], " [B] "),
        (row["Pronoun-offset"], " [P] ")
    ], key=lambda x: x[0], reverse=True)
    text = row["Text"]
    for offset, tag in to_be_inserted:
        text = text[:offset] + tag + text[offset:]
    return text


def tokenize(text, tokenizer):
    """Returns a list of tokens and the positions of A, B, and the pronoun."""
    entries = {}
    final_tokens = []
    for token in tokenizer.tokenize(text):
        if token in ("[A]", "[B]", "[P]"):
            entries[token] = len(final_tokens)
            continue
        final_tokens.append(token)
    return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


nlp = spacy.load('en')


def get_sentence(text, offset, token_after="[PRONOUN]"):
    """
    Extract a sentence containing a word at position offset by character and
    replace the word with token_after.
    output: Transformed sentence
            A word starting at offset
            A pos tag of the word.
            Default values if the word cannot be extracted.
    """
    doc = nlp(text)
    # idx: Character offset
    idx_begin = 0
    sent = None
    for token in doc:
        if token.sent_start:
            idx_begin = token.idx
        if token.idx == offset:
            sent = token.sent.string
            pos_tag = token.pos_
            idx_token = offset - idx_begin
            break
    # word_s = sent[idx_token:].split()
    # n = len(sent)
    if sent is None:
        # Default values
        sent_transformed = token_after
        token_before = "it"
        pos_tag = "PRON"
    else:
        token_before = token.string.strip()
        subtxt_transformed = re.sub("^" + token_before, token_after, sent[idx_token:])
        sent_transformed = sent[:idx_token] + subtxt_transformed
    # n_diff = len(sent_transformed) - n - len(token_after) + len(token_before)
    return sent_transformed, token_before, pos_tag


def generate_choices(text, offset, A, B, N=None, no_post_s=False):
    """
    Extract a sentence containing a pronoun at a offset position.
    Then convert the pronoun to A, B or "Neither A or B".
        3 choices.
        [Pronoun] likes something. ==>
          A likes something.
          B likes something.
          neigher A nor B likes something. (If N is None.)
          N likes something. (If N is not None.)
    text:  str
    offset: int
    A, B: Person's names. str
    N: nobody or something. str
    """
    sents = []
    text_pronoun, pronoun, pos_tag = get_sentence(text, offset)
    if pos_tag == "ADJ" or pronoun == "hers":
        _post = "'s"
    elif pronoun == "his":
        _post = "'s"
    else:
        _post = ""
    if no_post_s:
        _post = ""

    who_s = [A + _post, B + _post]
    if N is None:
        who_s += ["neither " + A + " nor " + B]
    else:
        who_s += [N + _post]
    sents.extend([re.sub("\[PRONOUN\]", who, text_pronoun) for who in who_s])

    return sents


def collate_fn(batch):
    """
    Pad the inputs sequence.
    """
    x_lst, y_lst = list(zip(*batch))
    xy_batch = [pad_sequence(x, batch_first=True) for x in zip(*x_lst)]
    xy_batch.append(torch.stack(y_lst, dim=0))
    return xy_batch


def collate_examples(batch, truncate_len=500):
    """Batch preparation.

    1. Pad the sequences
    2. Transform the target.
    """
    transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    # Offsets
    offsets = torch.stack([
        torch.LongTensor(x) for x in transposed[1]
    ], dim=0) + 1 # Account for the [CLS] token
    # Labels
    if len(transposed) == 2:
        return token_tensor, offsets, None
    one_hot_labels = torch.stack([
        torch.from_numpy(x.astype("uint8")) for x in transposed[2]
    ], dim=0)
    _, labels = one_hot_labels.max(dim=1)
    return token_tensor, offsets, labels


class GAPDataset(Dataset):
    """Custom GAP Dataset class"""
    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        if labeled:
            tmp = df[["A-coref", "B-coref"]].copy()
            tmp["Neither"] = ~(df["A-coref"] | df["B-coref"])
            self.y = tmp.values.astype("bool")
        # Extracts the tokens and offsets(positions of A, B, and P)
        self.offsets, self.tokens = [], []
        self.seq_len = []
        for _, row in df.iterrows():
            text = insert_tag(row)
            tokens, offsets = tokenize(text, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + tokens + ["[SEP]"]))
            self.seq_len.append(len(self.tokens[-1]))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx]  #, None

    def get_seq_len(self):
        return self.seq_len


class NLIDataset(Dataset):
    """
    NLI Dataset
    p_texts: Premise texts
    h_texts: Hypothesis texts
    tokenizer: BertTokenizer
    y      : Target sequence
    y_values: Class labels
    """
    def __init__(self, p_texts, h_texts, tokenizer,
                 y=None, y_values=None, max_len=100):
        if y is None:
            self.labels = torch.LongTensor([0] * len(p_texts))
        else:
            mapper = {label: i for i, label in enumerate(y_values)}
            self.labels = torch.LongTensor([mapper[v] for v in y])

        self.max_tokens = 0
        self.inputs = []
        self.seq_len = []
        for e, (p_text, h_text) in enumerate(zip(p_texts, h_texts)):
            p_tokens = tokenizer.tokenize(p_text)
            h_tokens = tokenizer.tokenize(h_text)
            _truncate_seq_pair(p_tokens, h_tokens, max_len - 3)
            p_len = len(p_tokens)
            h_len = len(h_tokens)

            tokens = ["[CLS]"] + p_tokens + ["[SEP]"] + h_tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * (p_len + 2) + [1] * (h_len + 1)
            input_mask = [1] * len(input_ids)
            self.inputs.append([torch.LongTensor(input_ids),
                                torch.LongTensor(segment_ids),
                                torch.LongTensor(input_mask)])
            self.seq_len.append(p_len + h_len + 3)
            self.max_tokens = max(self.seq_len[-1], self.max_tokens)
            if e < 1:
                print("tokens:", p_tokens)

        print("max_len:", self.max_tokens)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.inputs)

    def get_seq_len(self):
        return self.seq_len


def get_offsets_loader(config, train_df=None, test_df=None):
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model,
        do_lower_case=config.do_lower_case,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
    )
    # These tokens are not actually used, so we can assign arbitrary values.
    tokenizer.vocab['[A]'] = -1
    tokenizer.vocab['[B]'] = -1
    tokenizer.vocab['[P]'] = -1

    if train_df is not None:
        train_ds = GAPDataset(train_df, tokenizer)

        train_loader = DataLoader(
            train_ds,
            collate_fn=collate_examples,
            batch_size=config.train_batch_size,
            shuffle=True,
            drop_last=True
        )
        train_eval_loader = DataLoader(
            train_ds,
            collate_fn=collate_examples,
            batch_size=config.eval_batch_size,
            shuffle=False
        )
    else:
        train_loader = train_eval_loader = None

    if test_df is not None:
        test_ds = GAPDataset(test_df, tokenizer, labeled=False)

        test_loader = DataLoader(
            test_ds,
            collate_fn=collate_examples,
            batch_size=config.eval_batch_size,
            shuffle=False
        )
    else:
        test_loader = None

    return train_loader, train_eval_loader, test_loader


def get_nli_dataset(df, tokenizer, max_len, N=None, no_post_s=False, labeled=True):
    p_sents = df["Text"].repeat(3)
    h_texts = df.apply(lambda x: generate_choices(x["Text"], x["Pronoun-offset"],
                                                  x["A"], x["B"], N=N,
                                                  no_post_s=no_post_s),
                       axis=1)
    h_texts = sum(h_texts, [])
    if labeled:
        y_A = df["A-coref"].astype(int)
        y_B = df["B-coref"].astype(int)
        y_Neither = 1 - y_A - y_B
        labels = np.column_stack((y_A, y_B, y_Neither)).reshape(-1)
    else:
        labels = None
    return NLIDataset(p_sents, h_texts, tokenizer,
                      y=labels, y_values=(0, 1), max_len=max_len,)


def get_nli_loader(config,
                   train_df=None,
                   test_df=None):
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_model,
        do_lower_case=config.do_lower_case,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[PRONOUN]")
    )

    if train_df is not None:
        tr_ds = get_nli_dataset(train_df, tokenizer, config.max_len, N=config.N,
                                no_post_s=config.no_post_s)

        tr_loader = DataLoader(
            tr_ds,
            batch_size=config.train_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True
        )
        tr_eval_loader = DataLoader(
            tr_ds,
            batch_size=config.eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
    else:
        tr_loader = tr_eval_loader = None

    if test_df is not None:
        test_ds = get_nli_dataset(test_df, tokenizer, config.max_len, N=config.N,
                                  no_post_s=config.no_post_s, labeled=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=config.eval_batch_size,
            collate_fn=collate_fn,
            shuffle=False
        )
    else:
        test_loader = None
    return tr_loader, tr_eval_loader, test_loader


def get_pretrained_bert(modelname, num_hidden_layers=None):
    bert = BertModel.from_pretrained(modelname)
    if num_hidden_layers is None:
        return bert
    old_num_hidden_layers = bert.config.num_hidden_layers
    if num_hidden_layers < old_num_hidden_layers:
        # Only use the bottom n layers
        del bert.encoder.layer[num_hidden_layers:]
    elif num_hidden_layers > old_num_hidden_layers:
        # Add BertLayer(s)
        for i in range(old_num_hidden_layers, num_hidden_layers):
            bert.encoder.layer.add_module(str(i), BertLayer(bert.config))
    if num_hidden_layers != old_num_hidden_layers:
        bert.config.num_hidden_layers = num_hidden_layers
        bert.init_bert_weights(bert.pooler.dense)
    return bert


class BertCl(Module):
    def __init__(self, bert_model, n_bertlayers=1, dropout=0., num_labels=1,
                 no_pooler=False):
        super().__init__()
        if isinstance(bert_model, str):
            self.bert = get_pretrained_bert(bert_model, num_hidden_layers=n_bertlayers)
        elif isinstance(bert_model, BertModel):
            self.bert = bert_model
        self.dropout = Dropout(dropout)
        self.classifier = Linear(self.bert.config.hidden_size, num_labels)
        self.no_pooler = no_pooler

    def forward(self, input_ids, segment_ids, input_mask):
        encoded_layer, pooled_output = self.bert(input_ids, segment_ids, input_mask,
                                                 output_all_encoded_layers=False)
        if self.no_pooler:
            x = self.classifier(self.dropout(encoded_layer[:, 0]))
        else:
            x = self.classifier(self.dropout(pooled_output))
        return x

    def predict(self, dataloader, proba=True, device=device):
        preds = []
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch[:-1])
            with torch.no_grad():
                logits = self.forward(*batch)
                preds.append(logits.clone().cpu())
        preds = torch.cat(preds) if len(preds) > 1 else preds[0]
        if proba:
            preds = torch.sigmoid(preds)
        preds = preds.numpy()
        return preds


class BertClOffsets(Module):
    """The main model."""
    def __init__(self, bert_model, n_bertlayers=1, dropout=0., n_offsets=3):
        super().__init__()
        if isinstance(bert_model, str):
            self.bert = get_pretrained_bert(bert_model, num_hidden_layers=n_bertlayers)
        elif isinstance(bert_model, BertModel):
            self.bert = bert_model
        self.bert_hidden_size = self.bert.config.hidden_size
        self.dropout = Dropout(dropout)
        self.classifier = Linear(self.bert.config.hidden_size * n_offsets, n_offsets)

    def forward(self, token_tensor, offsets, label_id=None):
        bert_outputs, _ = self.bert(
            token_tensor, attention_mask=(token_tensor > 0).long(),
            token_type_ids=None, output_all_encoded_layers=False)
        extracted_outputs = bert_outputs.gather(
            1, offsets.unsqueeze(2).expand(-1, -1, bert_outputs.size(2))
        ).view(bert_outputs.size(0), -1)
        outputs = self.classifier(self.dropout(extracted_outputs))
        return outputs

    def predict(self, dataloader, proba=True, device=device):
        preds = []
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch[:2])
            with torch.no_grad():
                logits = self.forward(*batch)
                preds.append(logits.clone().cpu())
        preds = torch.cat(preds) if len(preds) > 1 else preds[0]
        if proba:
            preds = F.softmax(preds, dim=1)
        preds = preds.numpy()
        return preds


def run_epoch(model, dataloader, optimizer, criterion, gradient_accumulation_steps,
              verbose_step=10000, device=device):
    model.train()
    gradient_accumulation_steps = gradient_accumulation_steps
    t1 = time.time()
    tr_loss = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        label_ids = batch[-1]
        outputs = model(*batch[:-1])
        if criterion._get_name() == "BCEWithLogitsLoss":
            outputs = outputs[:, 0]
            label_ids = label_ids.float()
        loss = criterion(outputs, label_ids)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        tr_loss += loss.item()
        if (step + 1) % verbose_step == 0:
            loss_now = gradient_accumulation_steps * tr_loss / (step + 1)
            print(f'step:{step+1} loss:{loss_now:.7f} time:{time.time() - t1:.1f}s')
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()
    return gradient_accumulation_steps * tr_loss / (step + 1)


def get_gap_model(config, steps_per_epoch, device):
    model = BertClOffsets(config.bert_model, config.n_bertlayers, config.dropout)
    model.to(device)

    param_optimizer = list(model.named_parameters())

    if config.weight_decay:
        no_decay = ["bias", "gamma", "beta", "classifier"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer], "weight_decay": 0.0}

        ]

    t_total = int(
        steps_per_epoch / config.gradient_accumulation_steps * config.num_train_epochs)
    if config.optim == 'bertadam':
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.lr,
                             warmup=config.warmup_proportion,
                             t_total=t_total)
    elif config.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters,
                         lr=config.lr)
    else:
        print("--optim should be 'bertadam' or 'adam'.")
    return model, optimizer


def get_nli_model(config, steps_per_epoch, device):
    model = BertCl(config.bert_model, config.n_bertlayers, config.dropout)
    model.to(device)

    param_optimizer = list(model.named_parameters())

    if config.weight_decay:
        no_decay = ["bias", "gamma", "beta", "classifier"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer], "weight_decay": 0.0}

        ]

    t_total = int(
        steps_per_epoch / config.gradient_accumulation_steps * config.num_train_epochs)
    if config.optim == 'bertadam':
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.lr,
                             warmup=config.warmup_proportion,
                             t_total=t_total)
    elif config.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters,
                         lr=config.lr)
    else:
        print("--optim should be 'bertadam' or 'adam'.")
    return model, optimizer


# https://goodcode.io/articles/python-dict-object/
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
