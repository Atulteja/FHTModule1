import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
def f_riskscore(df, df_meta, n_std=4, parameters=None, means=None, stddevs=None, wts=None, dirns=None):
    if parameters is None or means is None or stddevs is None or wts is None or dirns is None:
        parameters, means, stddevs, wts, dirns = f_model_configurations(df_meta=None)

    df["risk score"] = 0
    for var, mean_vals in means.items():
        if wts[var] == 0 or var not in df.columns:
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
            print(f"Checking column: {var}")
            print(df[var].apply(type).value_counts())

            mask_bad = df[var].apply(lambda x: isinstance(x, str))
            print(df.loc[mask_bad, ["Participant", var]])
            df["point"] = ~df[var].between(th_low, th_high)
        else:
            print("Error")
        
        df["point"].replace({False: 0, True: 1 * wts[var]}, inplace=True)   
        
        df["risk score"] = df["risk score"] + df["point"]
    df = df.drop(["point"], axis=1)
    return df

def f_model_configurations(df_meta = None):
    if df_meta is None:
        df_meta = pd.read_excel(paths["Model_information"], sheet_name="Thresholds")

    parameters = df_meta["Parameter"].values.tolist()
    optimum_thresholds_mean = {k: float(g["Mean/cutoff"]) for k, g in df_meta.groupby("Parameter")}
    optimum_thresholds_std = {k: float(g["StdDev"]) for k, g in df_meta.groupby("Parameter")}
    wts = {k: int(g["Weights"]) for k, g in df_meta.groupby("Parameter")}
    dirns = {k: str(g["Faller_if"].to_list()[0]) for k, g in df_meta.groupby("Parameter")}

    return parameters, optimum_thresholds_mean, optimum_thresholds_std, wts, dirns

def f_thresholding_predict(df, use_ZM=False, use_QN=False, use_age_fall_history=False):
    config_key = f"ZM={int(use_ZM)}_QN={int(use_QN)}_AF={int(use_age_fall_history)}"
    cutoff = thresholds_by_config.get(config_key)
    
    if cutoff is None:
        raise ValueError(f"Threshold not defined for configuration: {config_key}")
    
    df = df.copy()
    df["prediction"] = (df["risk score"] >= cutoff).astype(int)
    return df, cutoff

def f_evaluate_predictions(df):
    df = f_thresholding_predict(df)
    y_pred = df["prediction"].values
    y_true = df["label"].values
    cm = confusion_matrix(y_true, y_pred)
    # labels = np.unique(y_true + y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    acc = (tp + tn) / (tn + fp + fn + tp)

    yidx = sens + spec - 1
    plr = sens / (1-spec)
    f1_ = f1_score(y_true, y_pred)
    aucroc_ = roc_auc_score(y_true, y_pred)
    return df, spec, sens, acc, yidx, plr, f1_, aucroc_

def f_generate_TARGET_dataset(cleaned_df: dict, use_ZM=False, use_QN=False, use_age_fall_history=False):
    
    has_ZM = 'ZM' in cleaned_df and cleaned_df['ZM'] is not None
    has_QN = 'Questionnaire' in cleaned_df and cleaned_df['Questionnaire'] is not None

    # Validate inputs
    if not (has_ZM or has_QN):
        st.warning("Please upload at least one data file to proceed.")
        return None
    
    df_ZM = None
    df_QN = None

    # Read ZM data
    if use_ZM and has_ZM:
        df_ZM = cleaned_df['ZM']
        print('ZM dataset - shape:', df_ZM.shape)

    # Read questionnaire data
    # if use_QN and use_age_fall_history and has_QN:
    #     questionnaire = QUESTIONNAIREDATA(
    #         cleaned_df['Questionnaire'],
    #         features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
    #     )
    # elif use_QN and has_QN and not use_age_fall_history:
    #     questionnaire = QUESTIONNAIREDATA(
    #         cleaned_df['Questionnaire'],
    #         features=['ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
    #     )
    # elif use_age_fall_history and has_QN and not use_QN:
    #     questionnaire = QUESTIONNAIREDATA(
    #         cleaned_df['Questionnaire'],
    #         features=['Fall_2', 'Age']
    #     )
    # else: 
    #     questionnaire = None

    if use_QN and has_QN:    
        df_QN = cleaned_df['Questionnaire']
        print('Questionnaire dataset - shape:', df_QN.shape)

    # Merge datasets
    target = DATASET()

    if df_ZM is not None and df_QN is not None:
        target.merge_datasets(df_ZM, df_QN, {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
    elif df_ZM is not None:
        target.dataset = df_ZM
    elif df_QN is not None:
        target.dataset = df_QN   
    else:
        st.warning("Please upload the selected data files to proceed.")

    return target

def f_fall_history_model(paths, num_follow_ups):
    target = f_generate_TARGET_dataset(paths, num_follow_ups)
    df = f_riskscore(target.dataset, paths["Model_information"])
    df, spec, sens, acc, yidx, plr, f1_, aucroc_ = f_evaluate_predictions(df)
    return df, spec, sens, acc, yidx, plr, f1_, aucroc_ 

if __name__ == "__main__":

    paths = {}
    paths['ZM'] = "//1TB/Dataset_Feb2025/TARGETZMParameters_All_20241015.xlsx"
    paths['Questionnaires'] = "/1TB/Dataset_Feb2025/TARGET 5 November 2024 dataset to SEC_051124 numeric n=2291.xlsx"
    paths['Prospective'] = "/1TB/Dataset_Feb2025/TARGET follow-up 18.02.2025 to SEC 26022025 numeric.xls"
    paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    df, spec, sens, acc, yidx, plr, f1_, aucroc_  = f_fall_history_model(paths, 4)



