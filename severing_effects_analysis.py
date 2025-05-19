'''
This file is adapted from the original ROME repository by Kevin Meng (2022)
URL: https://github.com/kmeng01/rome
Modified in 2025 to support modified severing effects analysis across a range of autoregressive Transformer models,
including GPT, LLaMA, Qwen, and DeepSeek.
'''

import os, re
import torch, numpy
import importlib, copy
import transformers
from collections import defaultdict
from util import nethook
from matplotlib import pyplot as plt
from experiments.restoration_effects_analysis import (
    ModelAndTokenizer,
    make_inputs,
    predict_from_input,
    decode_tokens,
    layername,
    find_token_range,
    trace_with_patch,
    plot_trace_heatmap,
    collect_embedding_std,
)
from util.globals import DATA_DIR
from dsets import KnownsDataset
import argparse

from huggingface_hub import login

# login(token="input your hugging face token")

model_success_map = {
    "gpt2-xl": list(range(100)),
    "meta-llama/Llama-3.2-1B": [
        0, 3, 6, 7, 10, 11, 12, 16, 17, 21, 25, 28, 29, 31, 32, 33, 34, 36, 37, 38,
        40, 41, 42, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 63, 66, 70, 72, 73,
        74, 78, 81, 86, 87, 88, 89, 90, 91, 93, 94, 96, 97, 98, 101, 102, 103, 104, 106,
        107, 108, 111, 112, 113, 114, 118, 119, 120, 123, 126, 127, 128, 129, 130, 132,
        133, 134, 135, 136, 140, 141, 143, 145, 148, 149, 150, 152, 156, 157, 160, 161,
        162, 163, 167, 168, 169, 170, 171, 172, 174
    ],
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": [
        2, 5, 7, 9, 20, 25, 29, 37, 50, 54, 59, 61, 78, 81, 93, 99, 104, 106, 117, 126,
        129, 130, 132, 134, 143, 161, 169, 170, 174, 176, 181, 184, 190, 194, 195, 200,
        204, 206, 207, 209, 210, 213, 216, 224, 227, 232, 237, 241, 243, 268, 281, 283,
        292, 293, 298, 302, 303, 314, 320, 321, 327, 334, 335, 337, 341, 347, 358, 371,
        378, 380, 382, 384, 390, 392, 406, 408, 410, 413, 417, 419, 423, 427, 431, 433,
        445, 446, 450, 451, 452, 455, 456, 460, 464, 465, 474, 488, 490, 496, 501, 502
    ],
    "Qwen/Qwen2.5-1.5B": [
        0, 3, 4, 5, 6, 7, 8, 10, 14, 15, 16, 18, 19, 20, 25, 26, 28, 29, 30, 32, 33, 34,
        36, 37, 38, 40, 41, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 60, 61,
        66, 68, 69, 70, 71, 72, 74, 78, 81, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96,
        97, 98, 101, 102, 103, 104, 105, 106, 112, 113, 114, 119, 120, 123, 125, 126,
        127, 128, 130, 131, 134, 135, 136, 140, 143, 148, 149, 150, 151, 155, 156, 157,
        159, 160, 161, 162, 163, 165
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="input model name")
args = parser.parse_args()

# model_name = "gpt2-xl"
model_name = args.model_name

success_list = model_success_map.get(model_name, [])

print(f"Model: {model_name}")
print(f"Success List: {success_list}")

mt = ModelAndTokenizer(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
print(f"Using noise level {noise_level}")


def trace_with_repatch(
        model,  # The model
        inp,  # A set of inputs
        states_to_patch,  # A list of (token index, layername) triples to restore
        states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
        answers_t,  # Answer probabilities to collect
        tokens_to_mix,  # Range of tokens to corrupt (begin, end)
        noise=0.1,  # Level of noise to add
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
                model,
                [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
                edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow_3(
        mt,
        prompt,
        subject,
        token_range=None,
        samples=10,
        noise=0.1,
        window=10,
        extra_token=0,
        disable_mlp=False,
        disable_attn=False,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    if token_range == "last_subject":
        token_range = [e_range[1] - 1]
    e_range = (e_range[0], e_range[1] + extra_token)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    differences = trace_important_states_3(
        mt.model,
        mt.num_layers,
        inp,
        e_range,
        answer_t,
        noise=noise,
        disable_mlp=disable_mlp,
        disable_attn=disable_attn,
        token_range=token_range,
    )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind="",
    )


def trace_important_states_3(
        model,
        num_layers,
        inp,
        e_range,
        answer_t,
        noise=0.1,
        disable_mlp=False,
        disable_attn=False,
        token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    zero_mlps = []
    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        zero_mlps = []
        if disable_mlp:
            zero_mlps = [
                (tnum, layername(model, L, "mlp")) for L in range(0, num_layers)
            ]
        if disable_attn:
            zero_mlps += [
                (tnum, layername(model, L, "attn")) for L in range(0, num_layers)
            ]
        row = []
        for layer in range(0, num_layers):
            r = trace_with_repatch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                zero_mlps,  # states_to_unpatch
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def calculate_last_subject(mt, prefix, entity, cache=None, token_range="last_subject"):
    def load_from_cache(filename):
        try:
            dat = numpy.load(f"{cache}/{filename}")
            return {
                k: v
                if not isinstance(v, numpy.ndarray)
                else str(v)
                if v.dtype.type is numpy.str_
                else torch.from_numpy(v)
                for k, v in dat.items()
            }
        except FileNotFoundError as e:
            return None

    no_attn_r = load_from_cache("no_attn_r.npz")
    uncached_no_attn_r = no_attn_r is None
    no_mlp_r = load_from_cache("no_mlp_r.npz")
    uncached_no_mlp_r = no_mlp_r is None
    ordinary_r = load_from_cache("ordinary.npz")
    uncached_ordinary_r = ordinary_r is None
    if uncached_no_attn_r:
        no_attn_r = calculate_hidden_flow_3(
            mt,
            prefix,
            entity,
            disable_attn=True,
            token_range=token_range,
            noise=noise_level,
        )
    if uncached_no_mlp_r:
        no_mlp_r = calculate_hidden_flow_3(
            mt,
            prefix,
            entity,
            disable_mlp=True,
            token_range=token_range,
            noise=noise_level,
        )
    if uncached_ordinary_r:
        ordinary_r = calculate_hidden_flow_3(
            mt, prefix, entity, token_range=token_range, noise=noise_level
        )
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        for u, r, filename in [
            (uncached_no_attn_r, no_attn_r, "no_attn_r.npz"),
            (uncached_no_mlp_r, no_mlp_r, "no_mlp_r.npz"),
            (uncached_ordinary_r, ordinary_r, "ordinary.npz"),
        ]:
            if u:
                numpy.savez(
                    f"{cache}/{filename}",
                    **{
                        k: v.cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in r.items()
                    },
                )
    if False:
        return (ordinary_r["scores"][0], no_attn_r["scores"][0], no_mlp_r["scores"][0])
    return (
        ordinary_r["scores"][0] - ordinary_r["low_score"],
        no_attn_r["scores"][0] - ordinary_r["low_score"],
        no_mlp_r["scores"][0] - ordinary_r["low_score"],
    )



import tqdm

knowns = KnownsDataset(DATA_DIR)
all_ordinary = []
all_no_attn = []
all_no_mlp = []

for i, knowledge in enumerate(tqdm.tqdm([knowns[success] for success in success_list])):
    ordinary, no_attn, no_mlp = calculate_last_subject(
        mt,
        knowledge["prompt"],
        knowledge["subject"],
        cache=f"results/ct_disable_attn/{model_name}/case_{i}",
    )
    all_ordinary.append(ordinary)
    all_no_attn.append(no_attn)
    all_no_mlp.append(no_mlp)
title = "Causal effect of states at the early site with Attn or MLP modules severed"

avg_ordinary = torch.stack(all_ordinary).mean(dim=0)
avg_no_attn = torch.stack(all_no_attn).mean(dim=0)
avg_no_mlp = torch.stack(all_no_mlp).mean(dim=0)

print([knowns[success] for success in success_list])
print(len([knowns[success] for success in success_list]))
print()
print('avg_ordinary = ', avg_ordinary)
print('avg_no_attn = ', avg_no_attn)
print('avg_no_mlp = ', avg_no_mlp)

save_dir = "severed"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, model_name.replace("/", "_") + ".txt")

output_lines = []

output_lines.append(f"Model: {model_name}")
output_lines.append(f"Number of samples: {len(success_list)}\n")

output_lines.append("avg_ordinary = " + str(avg_ordinary.tolist()))
output_lines.append("avg_no_attn = " + str(avg_no_attn.tolist()))
output_lines.append("avg_no_mlp = " + str(avg_no_mlp.tolist()))

with open(save_path, "w") as f:
    f.write("\n".join(output_lines))

print(f"Result >>> {save_path} <<< saved.")