import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy.special import expit

sys.path.insert(0, '/1TB/wYr_model') 

from wYr_model_original import f_generate_TARGET_dataset, f_model_configurations

def evaluate_weights_logreg(X, y, weights):
    """Evaluate weights using logistic regression approach"""
    weights_array = np.array([float(w) for w in weights])
    logits = np.dot(X, weights_array)
    probs = expit(logits)
    preds = (probs >= 0.5).astype(int)
    if len(np.unique(y)) < 2 or len(np.unique(preds)) < 2:
        return None
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    if (tn + fp) == 0 or (tp + fn) == 0:
        return None
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    acc = (tp + tn) / (tn + fp + fn + tp)
    yidx = sens + spec - 1
    plr = sens / (1 - spec) if spec < 1 else np.inf
    f1_ = f1_score(y, preds)
    aucroc_ = roc_auc_score(y, preds)
    return {
        'spec': spec,
        'sens': sens, 
        'acc': acc,
        'yidx': yidx,
        'plr': plr,
        'aucroc': aucroc_,
        'f1': f1_
    }

def comprehensive_evaluation_logreg(X, y, weights, n_bootstrap=200, sample_size=100):
    print("=== Full Dataset Evaluation ===")
    full_results = evaluate_weights_logreg(X, y, weights)
    if full_results is not None:
        print(f"Full dataset results:")
        for metric, value in full_results.items():
            print(f"  {metric}: {value:.4f}")
    else:
        print("Full dataset evaluation failed - check your data and weights")
        return None, None

    print(f"\n=== Bootstrap Evaluation (n={n_bootstrap}, sample_size={sample_size}) ===")
    bootstrap_results = {k: [] for k in ['spec', 'sens', 'acc', 'yidx', 'plr', 'aucroc', 'f1']}
    skipped = 0
    for i in range(n_bootstrap):
        if i % 50 == 0:
            print(f"Bootstrap iteration {i+1}/{n_bootstrap}")
        X_sample, y_sample = resample(X, y, replace=True, n_samples=sample_size, random_state=42+i, stratify=y)
        result = evaluate_weights_logreg(X_sample, y_sample, weights)
        if result is None:
            skipped += 1
            continue
        for metric, value in result.items():
            if not (np.isnan(value) or np.isinf(value)):
                bootstrap_results[metric].append(value)
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


if __name__ == "__main__":

    paths = {}
    paths['ZM'] = "/1TB/wYr_model/wYr_datasets/testing_ZM.xlsx"
    paths['Questionnaires'] = "/1TB/wYr_model/wYr_datasets/testing_questionnaire.xlsx"
    paths['Prospective'] = "/1TB/wYr_model/wYr_datasets/testing_followup.xlsx"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    target = f_generate_TARGET_dataset(paths, 4)
    df_base = target.dataset.copy()
    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()

    feature_cols = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
                              "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
                              "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed", "ICONFES_Score_Adjusted", "IPAQ_Cat", "Age", "Fall_2"]

    X = df_base[feature_cols].values
    y = df_base["label"].values
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # weights1 = [0.13113595,  0.12291758,  0.23433388, -0.08292311,  0.16714466, -0.47159192,
    #             -0.05774576, -0.02259974, -0.03646138,  0.23757977,  0.05706306,  0.13853639,
    #             0.05791303, -0.08366833,  0.0327659,   0.30539868]

#     weights1 = [ 0.31172939,  0.47683977, -0.06522961, -0.00094028,  0.42325239, -0.45483625,
#  -0.18240973, -0.08755279, -0.09742236,  0.59766768,  0.08300544,  0.07171237,
#   0.04566819, -0.03456036,  0.06704373,  0.38959013]
    
    weights1 = [ 2.65492989e-01,  2.59582957e-01, -8.12750011e-02, -1.28991379e-03,
  4.08542559e-01, -2.17515492e+00, -5.97735166e-02, -5.56250882e-02,
 -1.84528624e-01,  8.19725143e-01,  9.51819583e-02,  4.30253805e-02,
  8.76564002e-02, -4.49207567e-02,  1.44548912e-01,  4.68085356e-01]

    print("Starting comprehensive evaluation (logistic regression model)...")
    full_results, bootstrap_results = comprehensive_evaluation_logreg(
        X_scaled, y, weights1, n_bootstrap=1000, sample_size=100
    )

# Visualization
if bootstrap_results and bootstrap_results['f1']:
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