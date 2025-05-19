import numpy, os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

output_dir = "aie"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

arch_pairs = [
    ("ns3_r0_gpt2-xl", "GPT-2-XL"),
    ("ns3_r0_meta-llama_Llama-3.2-1B", "LLaMA-3.2-1B"),
    ("ns3_r0_Qwen_Qwen2.5-1.5B", "Qwen-2.5-1.5B"),
    ("ns3_r0_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-1.5B"),
]

max_le_layer_info = []
ATE = None

class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        return numpy.concatenate(self.d).mean(axis=0)

    def std(self):
        return numpy.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)

def read_knowlege(count=150, kind=None, arch="ns3_r0_gpt2-xl"):
    dirname = f"results/{arch}/causal_trace/cases/"
    kindcode = "" if not kind else f"_{kind}"
    (
        avg_fe,
        avg_ee,
        avg_le,
        avg_fa,
        avg_ea,
        avg_la,
        avg_hs,
        avg_ls,
        avg_fs,
        avg_fle,
        avg_fla,
    ) = [Avg() for _ in range(11)]

    valid_count = 0
    for i in range(count):
        if valid_count == 100:
            break
        try:
            data = numpy.load(f"{dirname}/knowledge_{i}{kindcode}.npz")
        except:
            continue
        if "correct_prediction" in data and not data["correct_prediction"]:
            continue
        valid_count += 1
        scores = data["scores"]
        first_e, first_a = data["subject_range"]
        last_e = first_a - 1
        last_a = len(scores) - 1

        avg_hs.add(data["high_score"])
        avg_ls.add(data["low_score"])
        avg_fs.add(scores.max())
        avg_fle.add(scores[last_e].max())
        avg_fla.add(scores[last_a].max())
        avg_fe.add(scores[first_e])
        avg_ee.add_all(scores[first_e + 1 : last_e])
        avg_le.add(scores[last_e])
        avg_fa.add(scores[first_a])
        avg_ea.add_all(scores[first_a + 1 : last_a])
        avg_la.add(scores[last_a])

    result = numpy.stack(
        [
            avg_fe.avg(),
            avg_ee.avg(),
            avg_le.avg(),
            avg_fa.avg(),
            avg_ea.avg(),
            avg_la.avg(),
        ]
    )
    result_std = numpy.stack(
        [
            avg_fe.std(),
            avg_ee.std(),
            avg_le.std(),
            avg_fa.std(),
            avg_ea.std(),
            avg_la.std(),
        ]
    )
    print("Average Total Effect", avg_hs.avg() - avg_ls.avg())
    print("corrupted value", avg_ls.avg())
    print("Best average indirect effect on last subject", avg_le.avg().max() - avg_ls.avg())
    print("Best average indirect effect on last token", avg_la.avg().max() - avg_ls.avg())
    print("Average best-fixed score", avg_fs.avg())
    print("Average best-fixed on last subject token score", avg_fle.avg())
    print("Average best-fixed on last word score", avg_fla.avg())
    print("Argmax at last subject token", numpy.argmax(avg_le.avg()))
    print("Max at last subject token", numpy.max(avg_le.avg()))
    print("Argmax at last prompt token", numpy.argmax(avg_la.avg()))
    print("Max at last prompt token", numpy.max(avg_la.avg()))
    print()
    print()
    return dict(
        low_score=avg_ls.avg(), result=result, result_std=result_std, size=avg_fe.size(),ate=avg_hs.avg() - avg_ls.avg()
    )

def plot_array(
    differences,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
    archname="GPT2-XL",
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()
    answer = "AIE"
    labels = [
        "First subject token",
        "Middle subject tokens",
        "Last subject token",
        "First subsequent token",
        "Further tokens",
        "Last token",
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)

    cmap_lookup = {
        None: ListedColormap(sns.light_palette("#FF7F0E", n_colors=256)),   # hidden: orange
        "attn": ListedColormap(sns.light_palette("#1F77B4", n_colors=256)), # attention: blue
        "mlp": ListedColormap(sns.light_palette("#2CA02C", n_colors=256)),  # mlp: green
    }

    h = ax.pcolor(
        differences,
        cmap=cmap_lookup[kind],
        vmin=low_score,
        vmax=high_score,
    )

    if title:
        ax.set_title(title)
    ax.invert_yaxis()


    num_layers = differences.shape[1]
    tick_step = 5
    xticks = [i + 0.5 for i in range(0, num_layers, tick_step)]
    xticklabels = list(range(0, num_layers, tick_step))

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_yticklabels(labels)

    ax.set_xlabel(f"Layer")


    cb = plt.colorbar(h)
    if answer:
        cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
    plt.show()

for arch, archname in arch_pairs:
    print(f"\nProcessing {archname}...")
    ATE = None

    aie_hidden = []
    aie_mlp = []
    aie_attn = []

    max_le_layer_hidden = None
    max_le_layer_attn = None
    max_le_layer_mlp = None

    the_count = 1208
    high_score = None

    for kind in [None, "mlp", "attn"]:
        d = read_knowlege(the_count, kind, arch)
        count = d["size"]
        low_score = d["low_score"]
        aie = d["result"][2] - low_score
        ATE = d["ate"]

        if kind is None:
            aie_hidden = aie
            max_le_layer_hidden = int(numpy.argmax(d["result"][2]))
        elif kind == "mlp":
            aie_mlp = aie
            max_le_layer_mlp = int(numpy.argmax(d["result"][2]))
        elif kind == "attn":
            aie_attn = aie
            max_le_layer_attn = int(numpy.argmax(d["result"][2]))

        what = {
            None: "Hidden state",
            "mlp": "MLP",
            "attn": "Attention",
        }[kind]

        title = f"AIE of {what}"
        kindcode = "" if kind is None else f"_{kind}"
        result = numpy.clip(d["result"] - low_score, 0, None)

        if kind not in ["mlp", "attn"]:
            high_score = result.max()

        plot_array(
            result,
            kind=kind,
            title=title,
            low_score=0.0,
            high_score=high_score,
            archname=archname,
            savepdf=f"figures/{archname}{kindcode}_restoration_effects.png",
        )

    print("AIE (hidden):", aie_hidden)
    print("AIE (Attn):", aie_attn)
    print("AIE (MLP):", aie_mlp)
    print("ATE :", ATE)

    lines = []
    lines.append(f"ATE = {ATE}")
    lines.append("aie_hidden = [" + ", ".join(f"{v}" for v in aie_hidden) + "]")
    lines.append("aie_attn = [" + ", ".join(f"{v}" for v in aie_attn) + "]")
    lines.append("aie_mlp = [" + ", ".join(f"{v}" for v in aie_mlp) + "]")

    with open(os.path.join(output_dir, f"{archname}.txt"), "w") as f:
        f.write("\n".join(lines))


    max_le_layer_info.append(
        f"{archname}: hidden = {max_le_layer_hidden}, attn = {max_le_layer_attn}, mlp = {max_le_layer_mlp}"
    )

with open(os.path.join(output_dir, "max_last_subject_layer.txt"), "w") as f:
    f.write("\n".join(max_le_layer_info))

print('complete')