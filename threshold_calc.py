import pandas as pd
from wYr_model import f_thresholding_predict, f_generate_TARGET_dataset
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

config = ["ZM=1_QN=1_AF=1", 
          "ZM=1_QN=1_AF=0", 
          "ZM=1_QN=0_AF=1", 
          "ZM=1_QN=0_AF=0", 
          "ZM=0_QN=1_AF=1", 
          "ZM=0_QN=1_AF=0", 
          "ZM=0_QN=0_AF=1"]

gait_keywords = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed"]

questionnaire_keywords = ["ICONFES_Score_Adjusted", "IPAQ_Cat"]

age_fall_history_keywords = ["Age", "Fall_2"]

def filter_weights(df_meta, use_ZM, use_QN, use_AF):
    if not (use_ZM or use_QN or use_AF):
        ValueError("At least one of the datasets must be used: ZM, QN, or Age/Fall History.")
    if not use_ZM:
        df_meta.loc[df_meta["Parameter"].str.contains('|'.join(gait_keywords), regex=True), "Weights"] = 0
    if not use_QN:
        df_meta.loc[df_meta["Parameter"].str.contains('|'.join(questionnaire_keywords), regex=True), "Weights"] = 0
    if not use_AF:
        df_meta.loc[df_meta["Parameter"].str.contains('|'.join(age_fall_history_keywords), regex=True), "Weights"] = 0
    
    return df_meta


def evaluate_thresholds(df, config_name, th_range):
    results = []
    for th in th_range:
        df_eval = f_thresholding_predict(df.copy(), cutoff=th)
        y_pred = df_eval["prediction"].values
        y_true = df_eval["label"].values

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0
        sens = tp / (tp + fn) if (tp + fn) else 0
        yidx = sens + spec - 1
        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = np.nan
        results.append({"threshold": th, "F1": f1, "AUC": auc, "YI": yidx})
    
    result_df = pd.DataFrame(results)
    best_idx = result_df[["F1", "AUC", "YI"]].mean(axis=1).idxmax()
    best_row = result_df.iloc[best_idx]
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(result_df["threshold"], result_df["F1"], label="F1 Score")
    plt.plot(result_df["threshold"], result_df["AUC"], label="AUC ROC")
    plt.plot(result_df["threshold"], result_df["YI"], label="Youden's Index")
    plt.axvline(best_row["threshold"], color='gray', linestyle='--', label=f'Best Threshold = {int(best_row["threshold"])}')
    plt.title(f"Threshold Tuning for {config_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"threshold_plot_{config_name}.png")
    plt.close()

    
    return best_row


if __name__ == "__main__":

    configs = {
        "ZM=1_QN=1_AF=0": (1, 1, 0,  range(0, 74)),
        "ZM=1_QN=0_AF=1": (1, 0, 1,  range(0, 135)),
        "ZM=1_QN=0_AF=0": (1, 0, 0,  range(0, 64)),
        "ZM=0_QN=1_AF=1": (0, 1, 1,  range(0, 81)),
        "ZM=0_QN=1_AF=0": (0, 1, 0,  range(0, 10)),
        "ZM=0_QN=0_AF=1": (0, 0, 1,  range(0, 71)),
    }

    paths = {}
    paths['ZM'] = "//1TB/Dataset_Feb2025/TARGETZMParameters_All_20241015.xlsx"
    paths['Questionnaires'] = "/1TB/Dataset_Feb2025/TARGET 5 November 2024 dataset to SEC_051124 numeric n=2291.xlsx"
    paths['Prospective'] = "/1TB/Dataset_Feb2025/TARGET follow-up 18.02.2025 to SEC 26022025 numeric.xls"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    best_thresholds = {}
    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
    for config_name, (use_ZM, use_QN, use_AF, th_range) in configs.items():
        print(f"Evaluating configuration: {config_name}")
        
        # Generate full dataset
        df_full = f_generate_TARGET_dataset(paths, num_followups=4).dataset
        
        # Load and filter metadata
        df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
        df_meta = filter_weights(df_meta.copy(), use_ZM, use_QN, use_AF)

        # Compute risk scores using the filtered metadata
        parameters = df_meta["Parameter"].tolist()
        means = dict(zip(df_meta["Parameter"], df_meta["Mean/cutoff"]))
        stds = dict(zip(df_meta["Parameter"], df_meta["StdDev"]))
        weights = dict(zip(df_meta["Parameter"], df_meta["Weights"]))
        dirns = dict(zip(df_meta["Parameter"], df_meta["Faller_if"]))

        # Risk scoring logic (mimicking f_riskscore but directly using filtered_meta)
        df_full["risk score"] = 0
        for var in parameters:
            if dirns[var] == "<":
                th = means[var] - 4 * stds[var]
                df_full["point"] = (df_full[var] < th).astype(int) * weights[var]
            elif dirns[var] == ">":
                th = means[var] + 4 * stds[var]
                df_full["point"] = (df_full[var] > th).astype(int) * weights[var]
            elif dirns[var] == "=":
                th = means[var] + 4 * stds[var]
                df_full["point"] = (df_full[var] == th).astype(int) * weights[var]
            elif dirns[var] == "><":
                th_low = max(0, means[var] - 4 * stds[var]) if "Var" in var else means[var] - 4 * stds[var]
                th_high = means[var] + 4 * stds[var]
                df_full["point"] = (~df_full[var].between(th_low, th_high)).astype(int) * weights[var]
            else:
                continue
            df_full["risk score"] += df_full["point"]
        df_full.drop(columns=["point"], inplace=True)

        # Evaluate metrics across threshold range
        best = evaluate_thresholds(df_full, config_name, th_range)
        best_thresholds[config_name] = int(best["threshold"])
        print(f"{config_name} → Optimal Threshold = {int(best['threshold'])} | F1={best['F1']:.3f} | AUC={best['AUC']:.3f} | YI={best['YI']:.3f}")


    print(best_thresholds)