import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

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

def f_thresholding_predict(df, cutoff=64):
    df["prediction"] = df["risk score"] >= cutoff
    df["prediction"].replace({True: 1, False: 0}, inplace=True)
    return df

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

def f_generate_TARGET_dataset(uploaded_files: dict, use_ZM=False, use_QN=False, use_age_fall_history=False):
    
    #Flags
    has_ZM = 'ZM' in uploaded_files and uploaded_files['ZM'] is not None
    has_QN = 'Questionnaire' in uploaded_files and uploaded_files['Questionnaire'] is not None

    # Validate inputs
    if not (has_ZM or has_QN):
        st.warning("Please upload at least one data file to proceed.")
        return None
    
    df_ZM = None
    df_QN = None

    # Read ZM data
    if use_ZM and has_ZM:
        zm = ZURICHMOVEDATA(uploaded_files['ZM'])
        zm.read_dataset()
        df_ZM = zm.dataset
        print('ZM dataset - shape:', df_ZM.shape)

    # Read questionnaire data
    if use_QN and use_age_fall_history and has_QN:
        questionnaire = QUESTIONNAIREDATA(
            uploaded_files['Questionnaire'],
            features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
        )
    elif use_QN and has_QN and not use_age_fall_history:
        questionnaire = QUESTIONNAIREDATA(
            uploaded_files['Questionnaire'],
            features=['ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
        )
    elif use_age_fall_history and has_QN and not use_QN:
        questionnaire = QUESTIONNAIREDATA(
            uploaded_files['Questionnaire'],
            features=['Fall_2', 'Age']
        )
    else: 
        questionnaire = None

    if questionnaire is not None:    
        questionnaire.read_dataset()
        df_QN = questionnaire.dataset
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

    # if num_followups>0 and len(paths['Prospective'])>0:
    #     # read prospective data
    #     followup = PROSPECTIVEDATA(paths['Prospective'])
    #     followup.read_dataset()
    #     followup.generate_labels(num_followups=num_followups)

    #     # construct the dataset (TARGET) - with followup information
    #     complete_dataset = DATASET()
    #     complete_dataset.merge_datasets(target.dataset, followup.labels,
    #                         {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
    #     complete_dataset.dataset.dropna(inplace=True)

    #     print("---------------------------")
    #     print(complete_dataset.dataset.label.value_counts())
    #     print("---------------------------")

    #     target = complete_dataset
    # else:
    #     print("Follow up information not incorporated")

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



