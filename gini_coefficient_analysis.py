import numpy as np

archnames = [
    "GPT-2-XL",
    "LLaMA-3.2-1B",
    "Qwen-2.5-1.5B",
    "DeepSeek-R1-Distill-Qwen-1.5B",
]

#Gini Coefficient
def gini(aie_values, total_effect):
    a = np.array(aie_values)

    # normalize
    a = (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-8)
    n = len(a)
    diff_sum = np.sum(np.abs(a[:, None] - a[None, :]))
    return diff_sum / (2 * n * np.sum(a))

results = []

for archname in archnames:
    try:
        with open(f"aie/{archname}.txt", "r") as f:
            lines = f.readlines()

        ATE = eval(lines[0].split("=")[1].strip())
        aie_hidden = eval(lines[1].split("=")[1].strip())
        aie_attn = eval(lines[2].split("=")[1].strip())
        aie_mlp = eval(lines[3].split("=")[1].strip())

        gini_hidden = gini(aie_hidden, ATE)
        gini_attn = gini(aie_attn, ATE)
        gini_mlp = gini(aie_mlp, ATE)

        result = (
            f"{archname} Gini Coefficients (TE: {ATE})\n"
            f"Hidden State: {gini_hidden}\n"
            f"Attention:    {gini_attn}\n"
            f"MLP:          {gini_mlp}\n\n\n"
        )
        results.append(result)
    except FileNotFoundError:
        results.append(f"{archname}: File Not Found Error\n")
    except Exception as e:
        results.append(f"{archname}: Exception - {e}\n")

with open("aie/gini_all_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("complete")
