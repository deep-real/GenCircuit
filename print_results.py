import os
import pickle
import argparse
import math
from tabulate import tabulate

"""
This script will print a table of the following form:
Method      | IOI (GPT) | IOI (QWen) | IOI (Gemma) | IOI (Llama) | MCQA (QWen) | MCQA (Gemma) | MCQA (Llama) | Arithmetic (Llama) | ARC-E (Gemma) | ARC-E (Llama) | ARC-C (Llama)
Random      |
Method 1    |
Method 2    |
...
"""

COL_MAPPING = {
    "ioi_gpt2": 0, "ioi_qwen2.5": 1, "ioi_gemma2": 2, "ioi_llama3": 3,
    "mcqa_qwen2.5": 4, "mcqa_gemma2": 5, "mcqa_llama3": 6,
    "arithmetic-addition_llama3": 7, "arithmetic-subtraction_llama3": 8,
    "arc-easy_gemma2": 9, "arc-easy_llama3": 10,
    "arc-challenge_llama3": 11
}


def area_under_curve(X, Y):
    print(X, Y)
    area_under = 0.
    for idx in range(len(X) - 1):
        x_1, x_2 = X[idx] / X[-1], X[idx+1] / X[-1]
        trapezoidal = (x_2 - x_1) * ((Y[idx+1] + Y[idx]) / 2)
        area_under += trapezoidal
    return area_under

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default='results')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--metric", type=str, default='cpr', choices=['cpr', 'cmd'])
    args = parser.parse_args()

    scores = {}
    for method_name in os.listdir(args.output_dir):
        if method_name not in scores:
            scores[method_name] = ["-"] * 12

        method_path = os.path.join(args.output_dir, method_name)
        for results_file in os.listdir(method_path):
            task, model, split, abs = os.path.splitext(results_file)[0].split("_")
            if f"{task}_{model}" not in COL_MAPPING:
                continue
            if split != args.split:
                continue
            
            abs = abs.split("abs-")[1]
            if args.metric == 'cpr' and abs == "True":
                # Check whether results exist for abs=False
                if os.path.exists(os.path.join(method_path, results_file.replace("abs-True", "abs-False"))):
                    continue
            elif args.metric == 'cmd' and abs == "False":
                # Check whether results exist for abs=True
                if os.path.exists(os.path.join(method_path, results_file.replace("abs-False", "abs-True"))):
                    continue

            results_path = os.path.join(method_path, results_file)
            with open(results_path, 'rb') as handle:
                results = pickle.load(handle)

            col_idx = COL_MAPPING[f"{task}_{model}"]
            scores[method_name][col_idx] = results

    table = []
    header = ["Method", "IOI (GPT)", "IOI (QWen)", "IOI (Gemma)", "IOI (Llama)", "MCQA (QWen)", "MCQA (Gemma)", "MCQA (Llama)",
              "Arithmetic (Llama)", "ARC-E (Gemma)", "ARC-E (Llama)", "ARC-C (Llama)"]
    table.append(header)

    for method_name in scores:
        row = [method_name]

        if args.metric == 'cpr':
            scorelist = [f"{s['area_under']:.2f}" if type(s) == dict else "-" for s in scores[method_name]]
        else:
            scorelist = [f"{s['area_from_1']:.2f}" if type(s) == dict else "-" for s in scores[method_name]]
        row.extend(scorelist)

        table.append(row)

    print(tabulate(table))
