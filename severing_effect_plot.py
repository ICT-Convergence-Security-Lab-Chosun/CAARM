import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
import os
import ast

arch_pairs = [
    ("gpt2-xl", 48),
    ("meta-llama/Llama-3.2-1B", 16),
    ("Qwen/Qwen2.5-1.5B", 28),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 28),
]

def load_severed_values(save_path):
    with open(save_path, "r", encoding="utf-8") as f:
        content = f.read()

    avg_ordinary = ast.literal_eval(content.split("avg_ordinary =")[1].split("avg_no_attn =")[0].strip())
    avg_no_attn = ast.literal_eval(content.split("avg_no_attn =")[1].split("avg_no_mlp =")[0].strip())
    avg_no_mlp = ast.literal_eval(content.split("avg_no_mlp =")[1].strip())

    return avg_ordinary, avg_no_attn, avg_no_mlp

save_dir = "severed"

for model_name, layer_num in arch_pairs:
    safe_model_name = model_name.split("/")[-1]
    save_path = os.path.join(save_dir, model_name.replace("/", "_") + ".txt")

    try:
        avg_ordinary, avg_no_attn, avg_no_mlp = load_severed_values(save_path)

        max_layer_to_plot = 16
        layer_num_to_plot = min(layer_num, max_layer_to_plot)

        avg_ordinary = avg_ordinary[:layer_num_to_plot]
        avg_no_attn = avg_no_attn[:layer_num_to_plot]
        avg_no_mlp = avg_no_mlp[:layer_num_to_plot]

        with plt.rc_context(rc={"font.family": "Times New Roman"}):
            # fig, ax = plt.subplots(1, figsize=(4, 3), dpi=300)
            fig, ax = plt.subplots(1, figsize=(6, 3.1), dpi=300)

            ax.bar(
                [i - 0.3 for i in range(layer_num_to_plot)],
                avg_ordinary,
                width=0.3,
                color="#FF7F0E",
                label="Effect of Hidden state",
            )
            ax.bar(
                [i for i in range(layer_num_to_plot)],
                avg_no_attn,
                width=0.3,
                color="#1F77B4",
                hatch="//",
                label="Attention severed",
            )
            ax.bar(
                [i + 0.3 for i in range(layer_num_to_plot)],
                avg_no_mlp,
                width=0.3,
                color="#2CA02C",
                hatch="\\\\",
                label="MLP severed",
            )

            ax.set_ylabel("AIE")
            ax.set_xlabel("Layer")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(None, max(0.025, 0.21))
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.show()

            savefig_path = f"figures/{safe_model_name}_severing_effects.png"
            os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
            fig.savefig(savefig_path, bbox_inches="tight")
            plt.close(fig)

    except FileNotFoundError:
        print(f"File Not Found: {save_path}")
    except Exception as e:
        print(f"{model_name} Exception: {e}")
