import numpy as np
import scipy.stats as stats
import pandas as pd

# V1 runs
acc_v1 = np.array([0.5391, 0.5550, 0.5497])
f1_v1 = np.array([0.4490, 0.4855, 0.4577])
bal_acc_v1 = np.array([0.6862, 0.7031, 0.6639])

# EffNet B4
acc_b4 = np.array([0.7364])
f1_b4 = np.array([0.6919])
bal_acc_b4 = np.array([0.7916])

# V1.1
acc_v1_1 = np.array([0.6576])
# V2
acc_v2 = np.array([0.6424])

def print_stats(name, arr):
    if len(arr) > 1:
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        ci = stats.t.interval(0.95, len(arr)-1, loc=mean, scale=stats.sem(arr))
        print(f"{name}: {mean:.4f} +- {std:.4f} | 95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
    else:
        print(f"{name}: {arr[0]:.4f} (single run)")

print("--- V1 ---")
print_stats("V1 Accuracy", acc_v1)
print_stats("V1 F1", f1_v1)
print_stats("V1 Bal Acc", bal_acc_v1)

print("--- Baselines ---")
print_stats("B4 Accuracy", acc_b4)
print_stats("V1.1 Accuracy", acc_v1_1)
print_stats("V2 Accuracy", acc_v2)

# Cohen's d against a constant (assuming B4 is the population mean or a constant for comparison)
def cohens_d(group1, val2):
    return (np.mean(group1) - val2) / np.std(group1, ddof=1)

print("\n--- Effect Sizes (Cohen's d vs EfficientNet-B4) ---")
print(f"V1 vs B4 Accuracy d = {cohens_d(acc_v1, acc_b4[0]):.2f}")
print(f"V1 vs B4 F1 d = {cohens_d(f1_v1, f1_b4[0]):.2f}")
print(f"V1 vs B4 Bal Acc d = {cohens_d(bal_acc_v1, bal_acc_b4[0]):.2f}")

