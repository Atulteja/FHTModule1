# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
# from sklearn.utils import resample

# sys.path.insert(0, '/1TB/wYr_model') 

# from wYr_model_original import f_generate_TARGET_dataset, f_model_configurations

# paths = {}
# paths['ZM'] = "//1TB/Dataset_Feb2025/Validation_ZM.xlsx"
# paths['Questionnaires'] = "/1TB/Dataset_Feb2025/Validation_questionnaire.xlsx"
# paths['Prospective'] = "/1TB/Dataset_Feb2025/Validation_followup.xlsx"
# paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

# parameters, mean_vals, std_vals, weights, dirns = f_model_configurations(paths["Model_information"])

# n_bootstrap = 1000
# sample_size = 70
# threshold = 64
# n_std = 4

# target = f_generate_TARGET_dataset(paths, 4)
# df_target = target.dataset.copy()

# def evaluate_weights(df, weights, parameters, mean_vals, std_vals, dirns, threshold=64, n_std=4):
#     """Evaluate the fitness of a set of weights by calculating the risk score and AUC"""
#     df = df.copy()
#     df["risk score"] = 0

#     for var in parameters:
#         mean = mean_vals[var]
#         std = std_vals[var]

#         if dirns[var] == "<":
#             th_low = mean - (n_std * std)
#             df["point"] = df[var] < th_low
#         elif dirns[var] == ">":
#             th_high = mean + (n_std * std)
#             df["point"] = df[var] > th_high
#         elif dirns[var] == "=":
#             th_high = mean + (n_std * std)
#             df["point"] = df[var] == th_high
#         elif dirns[var] == "><":
#             th_low = max(0, mean - (n_std * std)) if "Var" in var else mean - (n_std * std)
#             th_high = mean + (n_std * std)
#             df["point"] = ~df[var].between(th_low, th_high)
#         else:
#             df["point"] = 0

#         df["point"] = df["point"].astype(int) * int(weights[var])
#         df["risk score"] += df["point"]

#     df.drop(columns=["point"], inplace=True)
#     df["prediction"] = (df["risk score"] >= threshold).astype(int)

#     y_true = df["label"].values
#     y_pred = df["prediction"].values
#     cm = confusion_matrix(y_true, y_pred)

#     tn, fp, fn, tp = cm.ravel()
#     spec = tn / (tn + fp)
#     sens = tp / (tp + fn)
#     acc = (tp + tn) / (tn + fp + fn + tp)

#     yidx = sens + spec - 1
#     plr = sens / (1-spec)
#     f1_ = f1_score(y_true, y_pred)
#     aucroc_ = roc_auc_score(y_true, y_pred)

#     if len(np.unique(y_true)) < 2:
#         return None

#     return spec, sens, acc, yidx, plr, aucroc_, f1_

# skipped = 0
# f1s = []
# weights1 = {"Var_StepLengthLeft" : 1,
#             "Var_StepLengthRight": 9,
#             "Var_StepTimeL": 6,
#             "Var_StepTimeR": 4,
#             "Var_StrideTimeL": 14, 
#             "Var_StrideTimeR": 11, 
#             "Var_strLengthL": 9, 
#             "Var_strLengthR": 15, 
#             "Var_SwingTimeL": 5,
#             "Var_SwingTimeR": 8,
#             "Var_Dls": 6,
#             "Avg_GaitSpeed": 5,
#             "Fall_2": 67,
#             "Age": 7, 
#             "ICONFES_Score_Adjusted": 5,
#             "IPAQ_Cat": 6
# }
# for i in range(n_bootstrap):
#     if i % 100 == 0:
#         print(f"Bootstrap iteration {i+1}/{n_bootstrap}")
    
#     df_sample = resample(df_target, replace=True, n_samples=sample_size, random_state=12*i, stratify=df_target['label'])
#     result = evaluate_weights(df_sample, weights1, parameters, mean_vals, std_vals, dirns, threshold, n_std)

#     if result is None:
#         skipped += 1
#         continue

#     spec, sens, acc, yidx, plr, aucroc_, f1 = result
    
#     if f1 is not None:
#         f1s.append(f1)

# f1s = np.array(f1s)
# mean_f1 = np.mean(f1s)
# std_f1 = np.std(f1s)

# print(f"\nBootstrapped f1_score Evaluation:")
# print(f"Mean f1_score: {mean_f1:.4f}")
# print(f"Std f1_score:  {std_f1:.4f}")
# print(f"skipped: {skipped}")

# print(" ----------------------------------------------------------------------------------------------------------------------- ")

# print(f"yidx : {yidx:.4f}")
# print(f"plr : {plr:.4f}")
# print(f"AUCROC : {aucroc_:.4f}")
# print(f"spec : {spec:.4f}")
# print(f"sens : {sens:.4f}")
# print(f"acc : {acc:.4f}")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils import resample

sys.path.insert(0, '/1TB/wYr_model') 

from wYr_model_original import f_generate_TARGET_dataset, f_model_configurations

paths = {}
paths['ZM'] = "/1TB/wYr_model/wYr_datasets/testing_ZM.xlsx"
paths['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/testing_questionnaire.xlsx"
paths['Prospective'] = "/1TB/wYr_model/wYr_datasets/testing_followup.xlsx"
paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

parameters, mean_vals, std_vals, weights, dirns = f_model_configurations(paths["Model_information"])

# Improved parameters
n_bootstrap = 200  # Reduced but more reasonable
sample_size = 100  # Increased for more stable results
threshold = 64
n_std = 4

target = f_generate_TARGET_dataset(paths, 4)
df_target = target.dataset.copy()

print(f"Original dataset size: {len(df_target)}")
print(f"Class distribution: {df_target['label'].value_counts().to_dict()}")

def evaluate_weights(df, weights, parameters, mean_vals, std_vals, dirns, threshold=64, n_std=4):
    """Evaluate the fitness of a set of weights by calculating the risk score and metrics"""
    df = df.copy()
    df["risk score"] = 0

    for var in parameters:
        if var not in df.columns:
            print(f"Warning: Parameter {var} not found in dataset")
            continue
            
        mean = mean_vals[var]
        std = std_vals[var]

        if dirns[var] == "<":
            th_low = mean - (n_std * std)
            df["point"] = df[var] < th_low
        elif dirns[var] == ">":
            th_high = mean + (n_std * std)
            df["point"] = df[var] > th_high
        elif dirns[var] == "=":
            th_high = mean + (n_std * std)
            df["point"] = df[var] == th_high
        elif dirns[var] == "><":
            th_low = max(0, mean - (n_std * std)) if "Var" in var else mean - (n_std * std)
            th_high = mean + (n_std * std)
            df["point"] = ~df[var].between(th_low, th_high)
        else:
            df["point"] = 0

        df["point"] = df["point"].astype(int) * int(weights[var])
        df["risk score"] += df["point"]

    df.drop(columns=["point"], inplace=True)
    df["prediction"] = (df["risk score"] >= threshold).astype(int)

    y_true = df["label"].values
    y_pred = df["prediction"].values
    
    # Check if we have both classes
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return None
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Avoid division by zero
    if (tn + fp) == 0 or (tp + fn) == 0:
        return None
    
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    acc = (tp + tn) / (tn + fp + fn + tp)

    yidx = sens + spec - 1
    plr = sens / (1 - spec) if spec < 1 else np.inf
    f1_ = f1_score(y_true, y_pred)
    aucroc_ = roc_auc_score(y_true, y_pred)

    return {
        'spec': spec,
        'sens': sens, 
        'acc': acc,
        'yidx': yidx,
        'plr': plr,
        'aucroc': aucroc_,
        'f1': f1_
    }

def comprehensive_evaluation(df, weights, parameters, mean_vals, std_vals, dirns, 
                           threshold=64, n_std=4, n_bootstrap=200, sample_size=100):
    """Comprehensive evaluation with bootstrap sampling"""
    
    # First, evaluate on the full dataset
    print("=== Full Dataset Evaluation ===")
    full_results = evaluate_weights(df, weights, parameters, mean_vals, std_vals, dirns, threshold, n_std)
    
    if full_results is not None:
        print(f"Full dataset results:")
        for metric, value in full_results.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("Full dataset evaluation failed - check your data and weights")
        return None
    
    # Bootstrap evaluation
    print(f"\n=== Bootstrap Evaluation (n={n_bootstrap}, sample_size={sample_size}) ===")
    
    bootstrap_results = {
        'spec': [], 'sens': [], 'acc': [], 'yidx': [], 
        'plr': [], 'aucroc': [], 'f1': []
    }
    
    skipped = 0
    
    for i in range(n_bootstrap):
        if i % 50 == 0:
            print(f"Bootstrap iteration {i+1}/{n_bootstrap}")
        
        # Use stratified sampling to maintain class balance
        df_sample = resample(df, replace=True, n_samples=sample_size, 
                           random_state=42+i, stratify=df['label'])
        
        result = evaluate_weights(df_sample, weights, parameters, mean_vals, std_vals, dirns, threshold, n_std)

        if result is None:
            skipped += 1
            continue

        for metric, value in result.items():
            if not (np.isnan(value) or np.isinf(value)):
                bootstrap_results[metric].append(value)
    
    # Calculate bootstrap statistics
    print(f"\n=== Bootstrap Results Summary ===")
    print(f"Completed iterations: {n_bootstrap - skipped}")
    print(f"Skipped iterations: {skipped}")
    print(f"Skip rate: {skipped/n_bootstrap*100:.1f}%")
    
    if skipped/n_bootstrap > 0.5:
        print("WARNING: High skip rate suggests issues with sample size or class balance")
    
    print(f"\nBootstrap Statistics:")
    for metric, values in bootstrap_results.items():
        if len(values) > 0:
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            print(f"  {metric:>6}: {mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        else:
            print(f"  {metric:>6}: No valid results")
    
    return full_results, bootstrap_results

# Your optimized weights
weights1 = {
    "Var_StepLengthLeft": 38,
    "Var_StepLengthRight":0,
    "Var_StepTimeL": 12,
    "Var_StepTimeR": 7,
    "Var_StrideTimeL": 4, 
    "Var_StrideTimeR": 0, 
    "Var_strLengthL": 2, 
    "Var_strLengthR": 1, 
    "Var_SwingTimeL": 0,
    "Var_SwingTimeR": 11,
    "Var_Dls": 7,
    "Avg_GaitSpeed": 4,
    "Fall_2": 70,
    "Age": 9, 
    "ICONFES_Score_Adjusted": 0,
    "IPAQ_Cat": 0
}

weights2 = {
    "Var_StepLengthLeft": 3,
    "Var_StepLengthRight":3,
    "Var_StepTimeL": 3,
    "Var_StepTimeR": 3,
    "Var_StrideTimeL": 8, 
    "Var_StrideTimeR": 8, 
    "Var_strLengthL": 8, 
    "Var_strLengthR": 8, 
    "Var_SwingTimeL": 6,
    "Var_SwingTimeR": 6,
    "Var_Dls": 2,
    "Avg_GaitSpeed": 6,
    "Fall_2": 65,
    "Age": 6, 
    "ICONFES_Score_Adjusted": 4,
    "IPAQ_Cat": 6
}

print("Starting comprehensive evaluation...")
full_results, bootstrap_results = comprehensive_evaluation(
    df_target, weights2, parameters, mean_vals, std_vals, dirns, 
    threshold, n_std, n_bootstrap=1000, sample_size=70
)

print("Starting comprehensive evaluation for new set...")
full_results, bootstrap_results = comprehensive_evaluation(
    df_target, weights1, parameters, mean_vals, std_vals, dirns, 
    threshold, n_std, n_bootstrap=1000, sample_size=70
)

# Visualization
if bootstrap_results['f1']:
    plt.figure(figsize=(15, 10))
    
    metrics = ['f1', 'aucroc', 'sens', 'spec', 'acc', 'yidx']
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        if bootstrap_results[metric]:
            plt.hist(bootstrap_results[metric], bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(bootstrap_results[metric]), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(bootstrap_results[metric]):.3f}')
            plt.xlabel(metric.upper())
            plt.ylabel('Frequency')
            plt.title(f'{metric.upper()} Distribution')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No valid results', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()


