import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

thresholds_by_config = {
    "ZM=1_QN=1_AF=1": 64,
    "ZM=1_QN=1_AF=0": 45,
    "ZM=1_QN=0_AF=1": 63,
    "ZM=1_QN=0_AF=0": 54,
    "ZM=0_QN=1_AF=1": 11,
    "ZM=0_QN=1_AF=0": 0,
    "ZM=0_QN=0_AF=1": 7,
}

# calculate risk score
def f_riskscore(df, path_model_info, n_std=4):
    parameters, means, stddevs, wts, dirns = f_model_configurations(path_model_info)

    df["risk score"] = 0
    for var, mean_vals in means.items():
        if var not in df.columns:
            print(f"Warning: Variable '{var}' not found in dataset columns")
            continue

        if dirns[var] == "<":
            th_low = mean_vals - (n_std * stddevs[var])
            df["point"] = df[var] < th_low
        elif dirns[var] == ">":
            th_high = mean_vals + (n_std * stddevs[var])
            df["point"] = df[var] > th_high
        elif dirns[var] == "=":
            th_high = mean_vals + (n_std * stddevs[var])
            df["point"] = df[var] == th_high
        elif dirns[var] == "><":
            th_low = mean_vals - (n_std * stddevs[var])
            if "Var" in var and th_low < 0:
                th_low = 0
            th_high = mean_vals + (n_std * stddevs[var])
            df["point"] = ~df[var].between(th_low, th_high)
        else:
            print(f"Error: Unknown direction '{dirns[var]}' for variable '{var}'")
        
        df["point"].replace({False: 0, True: 1 * wts[var]}, inplace=True)   
        
        df["risk score"] = df["risk score"] + df["point"]
    df = df.drop(["point"], axis=1)
    return df

def f_model_configurations(path_model_info):
    df_meta = pd.read_excel(path_model_info, sheet_name="Thresholds")
    parameters = df_meta["Parameter"].values.tolist()
    
    # Fix the groupby aggregation issue
    optimum_thresholds_mean = {}
    optimum_thresholds_std = {}
    wts = {}
    dirns = {}
    
    for param in parameters:
        param_data = df_meta[df_meta["Parameter"] == param]
        optimum_thresholds_mean[param] = float(param_data["Mean/cutoff"].iloc[0])
        optimum_thresholds_std[param] = float(param_data["StdDev"].iloc[0])
        wts[param] = int(param_data["Weights"].iloc[0])
        dirns[param] = str(param_data["Faller_if"].iloc[0])

    return parameters, optimum_thresholds_mean, optimum_thresholds_std, wts, dirns

# def f_thresholding_predict(df, cutoff=64):
#     df["prediction"] = df["risk score"] >= cutoff
#     df["prediction"].replace({True: 1, False: 0}, inplace=True)
#     return df

def f_thresholding_predict(df, use_ZM=False, use_QN=False, use_age_fall_history=False):
    config_key = f"ZM={int(use_ZM)}_QN={int(use_QN)}_AF={int(use_age_fall_history)}"
    cutoff = thresholds_by_config.get(config_key)
    
    if cutoff is None:
        raise ValueError(f"Threshold not defined for configuration: {config_key}")
    
    df = df.copy()
    df["prediction"] = (df["risk score"] >= cutoff).astype(int)
    return df

def f_evaluate_predictions(df, use_ZM=False, use_QN=False, use_age_fall_history=False):
    df = f_thresholding_predict(df, use_ZM=True, use_QN=True, use_age_fall_history=True)
    y_pred = df["prediction"].values
    y_true = df["label"].values
    
    # Check if we have any positive predictions
    if y_pred.sum() == 0:
        print("Warning: No positive predictions made. All predictions are 0.")
        print(f"Risk score range: {df['risk score'].min()} to {df['risk score'].max()}")
        print(f"Using cutoff: 64")
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualize confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    
    # Handle division by zero cases
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc = (tp + tn) / (tn + fp + fn + tp)

    yidx = sens + spec - 1
    plr = sens / (1-spec) if spec < 1 else np.inf
    f1_ = f1_score(y_true, y_pred, zero_division=0)
    aucroc_ = roc_auc_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.5
    
    return df, spec, sens, acc, yidx, plr, f1_, aucroc_

def f_generate_TARGET_dataset(paths, num_followups=4, use_ZM=True, use_questionnaire=True, use_age_and_fall=True):
    print(f"Generating dataset with: ZM={use_ZM}, questionnaire={use_questionnaire}, age_and_fall={use_age_and_fall}")
    
    # Initialize components
    zm = None
    questionnaire = None
    
    # Read ZM data if needed
    if use_ZM:
        zm = ZURICHMOVEDATA(paths['ZM'])
        zm.read_dataset()
        print('ZM dataset shape:', zm.dataset.shape)

    # Read questionnaire data if needed
    if use_age_and_fall or use_questionnaire:
        features = []
        if use_age_and_fall:
            features.extend(['Fall_2', 'Age'])
        if use_questionnaire:
            features.extend(['ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted'])
        features = list(set(features))
        
        questionnaire = QUESTIONNAIREDATA(paths['Questionnaires'], features=features)
        questionnaire.read_dataset()
        print('Questionnaire dataset shape:', questionnaire.dataset.shape)

    # Check for missing participants if both datasets exist
    if zm is not None and questionnaire is not None:
        missing_info_participants = []
        for participant in zm.dataset["Participant"]:
            if participant not in list(questionnaire.dataset["Participant"]):
                missing_info_participants.append(participant)
        
        if len(missing_info_participants) > 0:
            print(f"Information missing for {len(missing_info_participants)} participants")

    # Construct the dataset (TARGET) - without followup information
    target = DATASET()
    
    if use_ZM and questionnaire is not None:
        target.merge_datasets(zm.dataset, questionnaire.dataset,
                              {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
    elif use_ZM and questionnaire is None:
        target.dataset = zm.dataset.copy()
    elif not use_ZM and questionnaire is not None:
        target.dataset = questionnaire.dataset.copy()
    else:
        print("Error: No data to construct TARGET dataset")
        return None

    print('TARGET dataset shape before dropna:', target.dataset.shape)
    target.dataset.dropna(inplace=True)
    print('TARGET dataset shape after dropna:', target.dataset.shape)

    # Add followup information if available
    if num_followups > 0 and len(paths['Prospective']) > 0:
        followup = PROSPECTIVEDATA(paths['Prospective'])
        followup.read_dataset()
        followup.generate_labels(num_followups=num_followups)

        complete_dataset = DATASET()
        complete_dataset.merge_datasets(target.dataset, followup.labels,
                            {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
        complete_dataset.dataset.dropna(inplace=True)

        print("Label distribution:")
        print(complete_dataset.dataset.label.value_counts())

        target = complete_dataset
    else:
        print("Follow up information not incorporated")

    return target

def f_fall_history_model(paths, num_follow_ups):
    results = {}

    combinations = [
        {"use_ZM": True,  "use_questionnaire": True,  "use_age_and_fall": True},
        {"use_ZM": True,  "use_questionnaire": True,  "use_age_and_fall": False},
        {"use_ZM": True,  "use_questionnaire": False, "use_age_and_fall": True},
        {"use_ZM": True,  "use_questionnaire": False, "use_age_and_fall": False},
        {"use_ZM": False, "use_questionnaire": True,  "use_age_and_fall": True},
        {"use_ZM": False, "use_questionnaire": True,  "use_age_and_fall": False},
        {"use_ZM": False, "use_questionnaire": False, "use_age_and_fall": True}
    ]

    for config in combinations:
        config_name = f"ZM={int(config['use_ZM'])}_QN={int(config['use_questionnaire'])}_AF={int(config['use_age_and_fall'])}"
        print(f"\n--- Processing configuration: {config_name} ---")
        
        try:
            target = f_generate_TARGET_dataset(
                paths,
                num_followups=num_follow_ups,
                use_ZM=config["use_ZM"],
                use_questionnaire=config["use_questionnaire"],
                use_age_and_fall=config["use_age_and_fall"]
            )

            if target is None:
                results[config_name] = {"error": "Failed to generate target dataset"}
                continue

            print(f"Dataset columns: {list(target.dataset.columns)}")
            
            df = f_riskscore(target.dataset, paths["Model_information"])
            df, spec, sens, acc, yidx, plr, f1_, aucroc_ = f_evaluate_predictions(df, use_ZM=config["use_ZM"], use_QN=config["use_questionnaire"], use_age_fall_history=config["use_age_and_fall"])

            results[config_name] = {
                "dataset_shape": target.dataset.shape,
                "risk_score_range": (df['risk score'].min(), df['risk score'].max()),
                "num_positive_labels": int(df['label'].sum()),
                "num_positive_predictions": int(df['prediction'].sum()),
                "specificity": float(spec),
                "sensitivity": float(sens),
                "accuracy": float(acc),
                "youden_index": float(yidx),
                "plr": float(plr) if not np.isinf(plr) else "inf",
                "f1_score": float(f1_),
                "auc_roc": float(aucroc_)
            }

        except Exception as e:
            print(f"Error in configuration {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[config_name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    paths = {}
    paths['ZM'] = "/1TB/Dataset_Feb2025/TARGETZMParameters_All_20241015.xlsx"
    paths['Questionnaires'] = "/1TB/Dataset_Feb2025/TARGET 5 November 2024 dataset to SEC_051124 numeric n=2291.xlsx"
    paths['Prospective'] = "/1TB/Dataset_Feb2025/TARGET follow-up 18.02.2025 to SEC 26022025 numeric.xls"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    results = f_fall_history_model(paths, 4)
    
    # Print results in a more readable format
    for config, result in results.items():
        print(f"\n{config}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            # print(f"  Dataset shape: {result['dataset_shape']}")
            # print(f"  Risk score range: {result['risk_score_range']}")
            print(f"  Positive labels: {result['num_positive_labels']}")
            print(f"  Positive predictions: {result['num_positive_predictions']}")
            print(f"  Sensitivity: {result['sensitivity']:.3f}")
            print(f"  Specificity: {result['specificity']:.3f}")
            print(f"  F1 Score: {result['f1_score']:.3f}")
            print(f"  AUC-ROC: {result['auc_roc']:.3f}")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  Youden Index: {result['youden_index']:.3f}")
            print(f"  Positive Likelihood Ratio: {result['plr']}")

