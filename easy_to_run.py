import subprocess


subprocess.run(["python", "experiments/restoration_effects_analysis.py", "--model_name", "gpt2-xl"])
subprocess.run(["python", "experiments/restoration_effects_analysis.py", "--model_name", "meta-llama/Llama-3.2-1B"])
subprocess.run(["python", "experiments/restoration_effects_analysis.py", "--model_name", "Qwen/Qwen2.5-1.5B"])
subprocess.run(["python", "experiments/restoration_effects_analysis.py", "--model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
subprocess.run(["python", "restoration_effect_plot.py"])


subprocess.run(["python", "severing_effects_analysis.py", "--model_name", "gpt2-xl"])
subprocess.run(["python", "severing_effects_analysis.py", "--model_name", "meta-llama/Llama-3.2-1B"])
subprocess.run(["python", "severing_effects_analysis.py", "--model_name", "Qwen/Qwen2.5-1.5B"])
subprocess.run(["python", "severing_effects_analysis.py", "--model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
subprocess.run(["python", "severing_effect_plot.py"])


subprocess.run(["python", "factual_prediction_analysis_gpt.py"])
subprocess.run(["python", "factual_prediction_analysis.py", "--model_name", "meta-llama/Llama-3.2-1B"])
subprocess.run(["python", "factual_prediction_analysis.py", "--model_name", "Qwen/Qwen2.5-1.5B"])
subprocess.run(["python", "factual_prediction_analysis.py", "--model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
subprocess.run(["python", "factual_prediction_plot.py"])


