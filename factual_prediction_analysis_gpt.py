'''
This file is adapted from the original dissecting_factual_predictions repository by Mor Geva (2023)
URL: https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions
'''

import csv
from ast import literal_eval
import functools
import json
import os
import random
import wget
import argparse

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

torch.set_grad_enabled(False)
tqdm.pandas()

# Utilities
from utils import (
    ModelAndTokenizer,
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_from_input,
)

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords0_ = stopwords.words('english')
stopwords0_ = {word: "" for word in stopwords0_}

# CounterFact Dataset
filename = "known_1000.json"

if not os.path.exists(filename):
    print(f"Downloading {filename}...")
    wget.download("https://rome.baulab.info/data/dsets/known_1000.json")
else:
    print(f"{filename} already exists. Skipping download.")

knowns_df = pd.read_json("known_1000.json")

model_success_map = {
    "gpt2-xl": list(range(100)),
}


model_name = "gpt2-xl"

success_list = model_success_map.get(model_name, [])

print(f"Model: {model_name}")
print(f"Success List: {success_list}")

mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=None,
)
mt.model.eval()

safe_model_name = model_name

# To block attention edges, we zero-out entries in the attention mask.
# To do this, we add a wrapper around the attention module, because
# the mask is passed as an additional argument, which could not be fetched
# with standard hooks before pytorch 2.0.
def set_block_attn_hooks(model, from_to_index_per_layer, opposite=False):
    """
    Only works on GPT2
    """

    def wrap_attn_forward(forward_fn, model_, from_to_index_, opposite_):
        @functools.wraps(forward_fn)
        def wrapper_fn(*args, **kwargs):
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(arg)
            for (k, v) in kwargs.items():
                new_kwargs[k] = v

            # print('args : ', args)

            hs = args[0]
            num_tokens = list(hs[0].size())[0]
            num_heads = model_.config.num_attention_heads

            if opposite_:
                attn_mask = torch.tril(torch.zeros((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 1
            else:
                attn_mask = torch.tril(torch.ones((num_tokens, num_tokens), dtype=torch.uint8))
                for s, t in from_to_index_:
                    attn_mask[s, t] = 0
            attn_mask = attn_mask.repeat(1, num_heads, 1, 1)

            attn_mask = attn_mask.to(dtype=model_.dtype)  # fp16 compatibility
            attn_mask = (1.0 - attn_mask) * torch.finfo(model_.dtype).min
            attn_mask = attn_mask.to(hs.device)

            new_kwargs["attention_mask"] = attn_mask

            return forward_fn(*new_args, **new_kwargs)

        return wrapper_fn

    hooks = []
    for i in from_to_index_per_layer.keys():
        hook = model.transformer.h[i].attn.forward
        model.transformer.h[i].attn.forward = wrap_attn_forward(model.transformer.h[i].attn.forward,
                                                                model, from_to_index_per_layer[i], opposite)
        hooks.append((i, hook))

    return hooks


def set_block_mlp_hooks(model, values_per_layer, coef_value=0):
    def change_values(values, coef_val):
        def hook(module, input, output):
            output[:, :, values] = coef_val

        return hook

    hooks = []
    for layer in range(model.config.n_layer):
        if layer in values_per_layer:
            values = values_per_layer[layer]
        else:
            values = []
        hooks.append(model.transformer.h[layer].mlp.c_fc.register_forward_hook(
            change_values(values, coef_value)
        ))

    return hooks

# Always remove your hooks, otherwise things will get messy.
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def remove_wrapper(model, hooks):
    for i, hook in hooks:
        model.transformer.h[i].attn.forward = hook


#################################
#Cache of hidden representations#
#################################

# create a cache of subject representations
print('mt.num_layers : ', mt.num_layers)
layers_to_cache = list(range(mt.num_layers + 1))
hs_cache = {}

for row_i, row in tqdm(knowns_df.iloc[success_list].iterrows()):

    prompt = row.prompt

    inp = make_inputs(mt.tokenizer, [prompt])
    output = mt.model(**inp, output_hidden_states=True)

    for layer in layers_to_cache:
        if (prompt, layer) not in hs_cache:
            hs_cache[(prompt, layer)] = []
        hs_cache[(prompt, layer)].append(output["hidden_states"][layer][0])

len(hs_cache)

# create a cache of subject representations

layers_to_cache = list(range(mt.num_layers))
subject_cache = {}

for row_i, row in tqdm(knowns_df.iloc[success_list].iterrows()):

    prompt = row.prompt
    subject = row.subject

    inp = make_inputs(mt.tokenizer, [prompt])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    e_range = [x for x in range(e_range[0], e_range[1])]

    output = mt.model(**inp, output_hidden_states=True)

    probs = torch.softmax(output["logits"][:, -1], dim=1)
    base_score, answer_t = torch.max(probs, dim=1)
    base_score = base_score.cpu().item()
    [answer] = decode_tokens(mt.tokenizer, answer_t)

    for layer in layers_to_cache:
        if (subject, layer) not in subject_cache:
            subject_cache[(subject, layer)] = []
        subject_cache[(subject, layer)].append(output["hidden_states"][layer + 1][0, e_range[-1]])

len(subject_cache)

E = mt.model.get_input_embeddings().weight.detach()
k = 500

##############################################
# applying knockouts to Attention/MLP module #
##############################################

all_mlp_dims = list(range(mt.model.config.n_embd * 4))
subject_repr_layer = mt.model.config.n_layer-1

num_block_layers = 5

records = []

for row_i, row in tqdm(knowns_df.iloc[success_list].iterrows()):

    prompt = row.prompt
    subject = row.subject
    inp = make_inputs(mt.tokenizer, [prompt])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    e_range = [x for x in range(e_range[0], e_range[1])]
    position = e_range[-1]

    output_ = mt.model(**inp, output_hidden_states=True)
    hs_ = output_["hidden_states"][subject_repr_layer + 1][0, position]
    projs_ = hs_.matmul(E.T).cpu().numpy()
    ind_ = np.argsort(-projs_)
    top_k_preds_ = [decode_tokens(mt.tokenizer, [i])[0] for i in ind_[:k]]

    for start_block_layer in range(subject_repr_layer):
        records.append({
            "example_index": row_i,
            "subject": subject,
            "layer": subject_repr_layer,
            "position": position,
            "block_layers": [],
            "block_module": "None",
            "start_block_layer": start_block_layer,
            "end_block_layer": -1,
            "num_block_layers": 0,
            "num_block_layers_": 0,
            "top_k_preds": top_k_preds_
        })

        end_block_layer = min(start_block_layer + num_block_layers + 1, subject_repr_layer)
        block_layers = [l for l in range(start_block_layer, end_block_layer)]
        for block_module in ["mlp", "attn"]:
            with torch.no_grad():
                if block_module == "mlp":
                    block_config = {layer_: all_mlp_dims for layer_ in block_layers}
                    block_mlp_hooks = set_block_mlp_hooks(mt.model, block_config)
                    output = mt.model(**inp, output_hidden_states=True)
                    remove_hooks(block_mlp_hooks)
                elif block_module == "attn":
                    block_config = {layer_: [] for layer_ in block_layers}
                    block_attn_hooks = set_block_attn_hooks(mt.model, block_config, opposite=True)
                    output = mt.model(**inp, output_hidden_states=True)
                    remove_wrapper(mt.model, block_attn_hooks)

            hs = output["hidden_states"][subject_repr_layer + 1][0, position]
            projs = hs.matmul(E.T).cpu().numpy()
            ind = np.argsort(-projs)

            records.append({
                "example_index": row_i,
                "subject": subject,
                "layer": subject_repr_layer,
                "position": position,
                "block_layers": block_layers,
                "block_module": block_module,
                "start_block_layer": start_block_layer,
                "end_block_layer": end_block_layer - 1,
                "num_block_layers": num_block_layers,
                "num_block_layers_": len(block_layers),
                "top_k_preds": [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            })

tmp = pd.DataFrame.from_records(records)
tmp.to_csv(f"csv/Factual_Prediction_Attention_MLP_{safe_model_name}.csv", index=False,
           quoting=csv.QUOTE_NONNUMERIC)

################
# Objects rate #
################

# This should be a path to a csv file with 2 columns and a header of column names "subject" and "paragraphs".
# Each entry should have (a) a subject (string) from the "knowns" data (knowns_df)
# and (b) paragraphs concatenated with space about the subject (a single string).

from sentence_transformers import SentenceTransformer, util
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


paragraphs_data_path = "wiki_subject_paragraphs.csv"
df_wiki = pd.read_csv(paragraphs_data_path)

# Tokenize, remove duplicate tokens, stopwords, and subwords.
df_wiki["context_tokenized_dedup"] = df_wiki["paragraphs"].progress_apply(
    lambda x: list(set(decode_tokens(mt.tokenizer, mt.tokenizer([x])['input_ids'][0])))
)
df_wiki["context_tokenized_dedup_len"] = df_wiki.context_tokenized_dedup.apply(lambda x: len(x))

df_wiki["context_tokenized_dedup_no-stopwords"] = df_wiki.context_tokenized_dedup.apply(
    lambda x: [
        y for y in x
        if y.strip() not in stopwords0_ and len(y.strip()) > 2
    ]
)
df_wiki["context_tokenized_dedup_no-stopwords_len"] = df_wiki["context_tokenized_dedup_no-stopwords"].apply(
    lambda x: len(x))


def get_preds_wiki_overlap_semantic(subject, top_preds, threshold=0.7):
    if len(top_preds) == 0:
        return -1

    wiki_row = df_wiki[df_wiki.subject == subject]
    if len(wiki_row) == 0:
        return -1

    wiki_toks = wiki_row.iloc[0]["context_tokenized_dedup_no-stopwords"]
    if not isinstance(wiki_toks, list) or len(wiki_toks) == 0:
        return -1

    preds_vecs = embedding_model.encode(top_preds, convert_to_tensor=True)
    wiki_vecs = embedding_model.encode(wiki_toks, convert_to_tensor=True)

    sim_matrix = util.cos_sim(preds_vecs, wiki_vecs)

    match_count = 0
    for i in range(len(top_preds)):
        max_sim = sim_matrix[i].max().item()
        if max_sim >= threshold:
            match_count += 1

    return match_count * 100.0 / len(top_preds)


import re

def is_english_word(word):
    return bool(re.fullmatch(r'[a-zA-Z]+', word))

#######################
#Evaluate objects rate#
#######################

tmp = pd.read_csv(f"csv/Factual_Prediction_Attention_MLP_{safe_model_name}.csv")

import ast
tmp["top_k_preds"] = tmp["top_k_preds"].apply(ast.literal_eval)

tmp["top_k_preds_clean"] = tmp.top_k_preds.progress_apply(lambda x: [
    y.strip() for y in x
    if y.strip().lower() not in stopwords0_ and len(y.strip()) > 2 and is_english_word(y.strip())
])

tmp["num_clean_tokens"] = tmp.top_k_preds_clean.progress_apply(lambda x: len(x))

m = 50  # evaluate the 50 top-scoring tokens

tmp["top_k_preds_in_context"] = tmp.progress_apply(
    lambda row: get_preds_wiki_overlap_semantic(row["subject"], row["top_k_preds_clean"][:m]),
    axis=1
    )

tmp.to_csv(f"csv/Factual_Prediction_{safe_model_name}.csv", index=False)

print(len(tmp[tmp.top_k_preds_in_context == -1]) * 100.0 / len(tmp))
print(tmp[tmp.top_k_preds_in_context > -1].subject.nunique())

print(f"✅ CSV save complete: Projection_result_{safe_model_name}.csv")
