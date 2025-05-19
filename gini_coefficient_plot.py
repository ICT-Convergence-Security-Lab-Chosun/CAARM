import matplotlib.pyplot as plt
import numpy as np
import re

plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(3,3))

target_models = {
    "GPT-2-XL",
    "LLaMA-3.2-1B",
    "Qwen-2.5-1.5B",
    "DeepSeek-R1-Distill-Qwen-1.5B"
}

archnames = []
gini_hidden = []
gini_attn = []
gini_mlp = []

with open("aie/gini_all_results.txt", "r", encoding="utf-8") as f:
    lines = [line for line in f if line.strip()]

for i in range(0, len(lines), 4):
    model_line = lines[i].strip()
    hidden_line = lines[i+1].strip()
    attn_line = lines[i+2].strip()
    mlp_line = lines[i+3].strip()

    model_name = model_line.split("Gini Coefficients")[0].strip()

    if model_name not in target_models:
        continue

    archnames.append(model_name)

    hidden_value = float(re.search(r"[-+]?\d*\.\d+|\d+", hidden_line).group())
    attn_value = float(re.search(r"[-+]?\d*\.\d+|\d+", attn_line).group())
    mlp_value = float(re.search(r"[-+]?\d*\.\d+|\d+", mlp_line).group())

    gini_hidden.append(hidden_value)
    gini_attn.append(attn_value)
    gini_mlp.append(mlp_value)

layer_dict = {}  # {model_name: (hidden, attn, mlp)}
with open("aie/max_last_subject_layer.txt", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        parts = line.strip().split(":")
        model = parts[0].strip()
        nums = list(map(int, re.findall(r'\d+', parts[1])))
        if len(nums) == 3:
            layer_dict[model] = tuple(nums)  # (hidden, attn, mlp)

x = np.arange(len(archnames))
width = 0.25

fig, ax = plt.subplots()

rects1 = ax.bar(x - width, gini_hidden, width, label='Hidden state',
                color='#FF7F0E', edgecolor='black', linewidth=1.5)
rects2 = ax.bar(x, gini_attn, width, label='Attention',
                color='#1F77B4', hatch='//', edgecolor='black', linewidth=1.5)
rects3 = ax.bar(x + width, gini_mlp, width, label='MLP',
                color='#2CA02C', hatch='\\\\', edgecolor='black', linewidth=1.5)

def label_layer(rects, layer_values, fontsize=12):
    for rect, val in zip(rects, layer_values):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width()/2.,
            height + 0.015,
            f'{val}',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight='bold',
            fontname='Times New Roman',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )

hidden_layers = [layer_dict.get(name, (None, None, None))[0] for name in archnames]
attn_layers = [layer_dict.get(name, (None, None, None))[1] for name in archnames]
mlp_layers = [layer_dict.get(name, (None, None, None))[2] for name in archnames]

pretty_names = []
for name in archnames:
    if "DeepSeek" in name:
        pretty_names.append(name.replace("DeepSeek-R1-", "DeepSeek-R1\n"))
    elif "Qwen-2.5" in name:
        pretty_names.append(name.replace("Qwen-2.5-", "Qwen-2.5\n"))
    elif "LLaMA-3.2" in name:
        pretty_names.append(name.replace("LLaMA-3.2-", "LLaMA-3.2\n"))
    else:
        pretty_names.append(name)


label_layer(rects1, hidden_layers)
label_layer(rects2, attn_layers)
label_layer(rects3, mlp_layers)

ax.set_ylabel('Gini Coefficient', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(pretty_names, fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1)


fig.tight_layout()

plt.savefig("figures/Gini_Coefficient.png", dpi=300, bbox_inches='tight')

plt.show()
