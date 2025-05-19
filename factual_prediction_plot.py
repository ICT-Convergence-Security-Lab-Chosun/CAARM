from os.path import split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.ticker as mticker

sns.set(context="notebook",
       rc={"font.size":16,
           "axes.titlesize":16,
           "axes.labelsize":16,
           "xtick.labelsize": 16.0,
           "ytick.labelsize": 16.0,
           "legend.fontsize": 16.0})

palette = ["gray", "#1F77B4", "#2CA02C"]
sns.set_theme(style='whitegrid', rc={
    "grid.color": "gray",
    "grid.alpha": 0.1,
    "grid.linewidth": 0.5
})

filenames = [
    "Factual_Prediction_gpt2-xl.csv",
    "Factual_Prediction_Llama-3.2-1B.csv",
    "Factual_Prediction_Qwen2.5-1.5B.csv",
    "Factual_Prediction_DeepSeek-R1-Distill-Qwen-1.5B.csv"
]

for filename in filenames:
    base_name = filename.removesuffix(".csv")
    print(base_name)

    # Load
    tmp = pd.read_csv(f"csv/{filename}")

    plt.rcParams["font.family"] = "Times New Roman"

    tmp["desc_short"] = tmp[['block_module', 'num_block_layers']].apply(tuple, axis=1)

    tmp["desc_short_"] = tmp.desc_short.apply(
        lambda x: {'mlp': "MLP Knockout",
                   'attn': "Attention Knockout"}.get(x[0], "No intervention")
    )
    tmp["start_block_layer_1"] = tmp.start_block_layer.apply(lambda x: x+1)

    # plt.figure(figsize=(4,2))
    plt.figure(figsize=(6,3))
    tmp_ = tmp[(tmp.top_k_preds_in_context > -1) & (tmp.num_block_layers.isin([0, 5]))]

    order = ["No intervention", "Attention Knockout", "MLP Knockout"]

    ax = sns.lineplot(data=tmp_,
                      x="start_block_layer", y="top_k_preds_in_context",
                      hue="desc_short_",
                      style="desc_short_",
                      hue_order=order,
                      style_order=order,
                      palette=palette[:3],
                      dashes=True,
                      linewidth=2,
                      markers=False,
                      errorbar=None
                     )
    ax.legend_.set_title("")

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)


    ax.set_ylabel(f"Objects Rate")
    ax.set_xlabel("Intervention layers")
    # ax.get_legend().remove()

    all_layers = sorted(tmp_["start_block_layer"].unique())
    max_layer = max(all_layers)

    tick_groups = [(i, i+5) for i in range(0, max_layer+1, 5)]

    tick_groups = [(start, min(end, max_layer + 1)) for start, end in tick_groups]

    xticks = [start for start, end in tick_groups]
    xticklabels = [f"{start}-{end-1}" for start, end in tick_groups]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    all_layers = sorted(tmp_["start_block_layer"].unique())
    ax.set_xticks(all_layers)

    plt.tight_layout()
    plt.savefig(f"figures/{base_name}.png")
    plt.show()