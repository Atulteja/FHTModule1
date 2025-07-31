import sys
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, '/1TB/wYr_model')

import matplotlib.pyplot as plt
import seaborn as sns
from wYr_model import f_model_configurations, f_riskscore, f_thresholding_predict, f_evaluate_predictions, f_generate_TARGET_dataset
from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

st.title("Fall Risk Prediction Tool")  

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

st.write("Welcome to the Fall Risk Prediction Tool — a clinical decision support app designed to help assess fall risk based on patient data.")
st.write("This tool allows healthcare professionals to upload patient data files, which are then processed to calculate a risk score and provide predictions regarding fall risk.")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

st.header("How to Use the Tool")
st.write("1. **Select Data Combination**: Select the type of data you want to combine for risk assessment. You can choose from ZurichMOVE data, questionnaire data and age and fall history data")
st.write("2. **Upload Data Files**: Upload the relevant data files in .csv or .xlsx format. The system will process these files to extract necessary information for risk assessment.")
st.write("3. **View Results**: After processing, a csv file will be generated containing the risk scores and predictions based on the uploaded data.")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

st.header("Data Selection and Upload")
st.write("To begin, please select the type of data you want to use for risk assessment. You can select one or more data types to combine for a comprehensive risk assessment:")
use_ZM = st.checkbox("Gait Data", value=True, key="zurichmove")
use_QN = st.checkbox("Questionnaire Data", value=True, key="questionnaire")
use_age_fall_history = st.checkbox("Age and Fall History Data", value=True, key="age_fall_history")

if not (use_ZM or use_QN or use_age_fall_history):
    st.warning("Please select at least one data type to proceed.")
    st.stop()

df_meta = pd.read_excel("/1TB/wYr_model/wYr_thresholds.xlsx", sheet_name="Thresholds")

if not (use_ZM or use_QN or use_age_fall_history):
    st.warning("Please select at least one data type to proceed with the risk assessment.")
if not use_ZM:
    gait_keywords = ["Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed"]
    df_meta.loc[df_meta["Parameter"].str.contains('|'.join(gait_keywords), regex=True), "Weights"] = 0
if not use_QN:
    questionnaire_keywords = ["ICONFES_Score_Adjusted", "IPAQ_Cat", "MOCA_Score_Adjusted"]
    df_meta.loc[df_meta["Parameter"].str.contains('|'.join(questionnaire_keywords), regex=True), "Weights"] = 0
if not use_age_fall_history:
    age_fall_history_keywords = ["Age", "Fall_2"]
    df_meta.loc[df_meta["Parameter"].str.contains('|'.join(age_fall_history_keywords), regex=True), "Weights"] = 0

parameters, means, stddevs, wts, dirns = f_model_configurations(df_meta=df_meta)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

st.header("Upload Your Data Files")
ZM_file = st.file_uploader("Upload your Gait data file here", type=["csv", "xlsx"], accept_multiple_files=False, key="zm_file_uploader") if use_ZM else None
QN_file = st.file_uploader("Upload your Questionnaire data file here", type=["csv", "xlsx"], accept_multiple_files=False, key = "qn_file_uploader") if use_QN or use_age_fall_history else None

uploaded_files = {
    "ZM" : ZM_file,
    "Questionnaire": QN_file
}

target = None
try:
    target = f_generate_TARGET_dataset(uploaded_files, use_ZM=use_ZM, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
except ValueError as e:
    st.warning(str(e))
    st.stop()

st.header("Processing and Results")
st.write(target)
st.write("Once uploaded, the system will process the data and calculate a risk score and provide predictions based on the provided data.")
st.write("You can then download the updated file for further review or analysis.")

if target is not None:
    missing_cols = []

    # Check expected questionnaire columns
    if use_QN or use_age_fall_history:
        expected_qn_cols = ["Participant", "ICONFES_Score_Adjusted", "IPAQ_Cat", "MOCA_Score_Adjusted"]
        for col in expected_qn_cols:
            if col not in target.dataset.columns:
                missing_cols.append(col)

    # Check expected gait columns
    if use_ZM:
        expected_gait_cols = ["Participant", "Var_StepLengthLeft", "Var_StepLengthRight", "Var_StepTimeL", "Var_StepTimeR", 
                              "Var_StrideTimeL", "Var_StrideTimeR", "Var_strLengthL", "Var_strLengthR", 
                              "Var_SwingTimeL", "Var_SwingTimeR", "Var_Dls", "Avg_GaitSpeed"]
        for col in expected_gait_cols:
            if col not in target.dataset.columns:
                missing_cols.append(col)

    # Check expected age & fall history columns
    if use_age_fall_history:
        expected_af_cols = ["Age", "Fall_2"]
        for col in expected_af_cols:
            if col not in target.dataset.columns:
                missing_cols.append(col)

    if missing_cols:
        st.info(f"The following expected columns were not found in the uploaded data: {', '.join(missing_cols)}")
        for col in missing_cols:
            df_meta.loc[df_meta["Parameter"] == col, "Weights"] = 0

if target is not None:
    df_processed = f_riskscore(target.dataset, df_meta, parameters=parameters, means=means, stddevs=stddevs, wts=wts, dirns=dirns)
    df_processed, cutoff = f_thresholding_predict(df_processed, use_ZM=use_ZM, use_QN=use_QN, use_age_fall_history=use_age_fall_history)

    active_params = df_meta[df_meta["Weights"] > 0]["Parameter"].tolist()
    keep_cols = ["Participant", 'risk score', 'prediction'] + active_params
    keep_cols = [col for col in keep_cols if col in df_processed.columns]

    df_final = df_processed[keep_cols]

    st.write("Processed Data:")
    st.write("Threshold for risk score:", cutoff)
    st.dataframe(df_final.head())

    csv_file = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed Data", data=csv_file, file_name="processed_data.csv", mime="text/csv")



def plot_bar(metrics_dict, config_label):
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color='skyblue')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title(f"Evaluation Metrics for {config_label}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig

df_eval = pd.read_excel("/1TB/wYr_model/Evaluation_metrics.xlsx", sheet_name="Sheet1")

config_key = f"ZM={int(use_ZM)}_QN={int(use_QN)}_AF={int(use_age_fall_history)}"

row = df_eval[df_eval["Config"] == config_key]

selected_metrics = ["Specificity", "Sensitivity", "Accuracy", "F1", "AUC-ROC"]
if not row.empty:
    metrics = row[selected_metrics].iloc[0].to_dict()

    st.subheader("Evaluation Metrics Bar Chart")
    fig = plot_bar(metrics, config_key)
    st.pyplot(fig)
else:
    st.info("No evaluation metrics available for this combination.")






