import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

# calculate risk score
def f_riskscore(df, path_model_info, n_std=4):
    parameters, means, stddevs, wts, dirns = f_model_configurations(path_model_info)

    df["risk score"] = 0
    for var, mean_vals in means.items():
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

def f_model_configurations(path_model_info):
    df_meta = pd.read_excel(paths['Model_information'], sheet_name="Thresholds")
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

def f_generate_TARGET_dataset(paths, num_followups=4):
    # read ZM data
    zm = ZURICHMOVEDATA(paths['ZM'])
    zm.read_dataset()
    print('zm dataset - shape:', zm.dataset.shape)

    # read questionnaire data
    questionnaire = QUESTIONNAIREDATA(paths['Questionnaires'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted'])
    questionnaire.read_dataset()
    print('questionnaire dataset - shape:', questionnaire.dataset.shape)

    missing_info_participants = []
    for participant in zm.dataset["Participant"]:
        if participant not in list(questionnaire.dataset["Participant"]):
            missing_info_participants.append(participant)
    
    if len(missing_info_participants)>0:
        print("Information missing for following participants:")
        for i in range(len(missing_info_participants)):
            print(missing_info_participants[i])

    # construct the dataset (TARGET) - without followup information
    target = DATASET()
    target.merge_datasets(zm.dataset, questionnaire.dataset,
                        {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
    print('TARGET dataset - shape:', target.dataset.shape)
    target.dataset.dropna(inplace=True)
    print('TARGET dataset - shape:', target.dataset.shape)

    if num_followups>0 and len(paths['Prospective'])>0:
        # read prospective data
        followup = PROSPECTIVEDATA(paths['Prospective'])
        followup.read_dataset()
        followup.generate_labels(num_followups=num_followups)

        # construct the dataset (TARGET) - with followup information
        complete_dataset = DATASET()
        complete_dataset.merge_datasets(target.dataset, followup.labels,
                            {'first_dataset': 'Participant', 'second_dataset': 'Participant'})
        complete_dataset.dataset.dropna(inplace=True)

        print("---------------------------")
        print(complete_dataset.dataset.label.value_counts())
        print("---------------------------")

        target = complete_dataset
    else:
        print("Follow up information not incorporated")

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

print(spec, sens, acc, yidx, plr, f1_, aucroc_)