import sys
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, '/1TB/wYr_model')

import matplotlib.pyplot as plt
import seaborn as sns
from wYr_model import f_model_configurations, f_riskscore, f_thresholding_predict, f_evaluate_predictions, f_generate_TARGET_dataset
from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET
from safetech_data_app_version import combine_uploaded_booster_files, combine_main_study_files, read_qn_file, read_zm_file


def process_risk_score(target, df_meta, parameters, means, stddevs, wts, dirns, 
                       use_QN=False, use_ZM=False, use_age_fall_history=False, 
                       show_info=True):   
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
            expected_gait_cols = [
                "Participant", "Var_StepLengthLeft", "Var_StepLengthRight", 
                "Var_StepTimeL", "Var_StepTimeR", "Var_StrideTimeL", "Var_StrideTimeR", 
                "Var_strLengthL", "Var_strLengthR", "Var_SwingTimeL", "Var_SwingTimeR", 
                "Var_Dls", "Avg_GaitSpeed"
            ]
            for col in expected_gait_cols:
                if col not in target.dataset.columns:
                    missing_cols.append(col)
        
        # Check expected age & fall history columns
        if use_age_fall_history:
            expected_af_cols = ["Age", "Fall_2"]
            for col in expected_af_cols:
                if col not in target.dataset.columns:
                    missing_cols.append(col)
        
        # Handle missing columns
        if missing_cols:
            if show_info:
                st.info(f"The following expected columns were not found in the uploaded data: {', '.join(missing_cols)}")
            
            # Set weights to 0 for missing columns
            for col in missing_cols:
                df_meta.loc[df_meta["Parameter"] == col, "Weights"] = 0
        
        # Process risk score
        df_processed = f_riskscore(target.dataset, df_meta, 
                                 parameters=parameters, means=means, 
                                 stddevs=stddevs, wts=wts, dirns=dirns)
        
        # Apply thresholding and prediction
        df_processed, cutoff = f_thresholding_predict(df_processed, 
                                                    use_ZM=use_ZM, 
                                                    use_QN=use_QN, 
                                                    use_age_fall_history=use_age_fall_history)
        
        # Select final columns
        active_params = df_meta[df_meta["Weights"] > 0]["Parameter"].tolist()
        keep_cols = ["Participant", 'risk score', 'prediction'] + active_params
        keep_cols = [col for col in keep_cols if col in df_processed.columns]
        df_final = df_processed[keep_cols]
        
        return df_final
    
    else:
        return None
    
def create_risk_summary(df, prediction_col="prediction", high_risk_value=1, low_risk_value=0):

    # Count high and low risk participants
    high_risk_count = (df[prediction_col] == high_risk_value).sum()
    low_risk_count = (df[prediction_col] == low_risk_value).sum()
    total = high_risk_count + low_risk_count
    
    # Compute percentages (handle division by zero)
    high_risk_pct = (high_risk_count / total) * 100 if total > 0 else 0
    low_risk_pct = (low_risk_count / total) * 100 if total > 0 else 0
    
    # Build summary DataFrame
    risk_summary_df = pd.DataFrame({
        "High Risk": [high_risk_count, f"{high_risk_pct:.2f}%"],
        "Low Risk": [low_risk_count, f"{low_risk_pct:.2f}%"]
    }, index=["Number of Participants", "Percentage"])
    
    return risk_summary_df

def create_summary_excel(summary_df, duplicates_matrix_df, duplicated_rows_df, 
                        risk_summary_df, missing_matrix_df, is_booster=False):
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write main summary first
        summary_df.to_excel(writer, sheet_name='Summary', index=True)
        
        if is_booster:
            # Calculate starting rows for additional data
            startrow2 = len(summary_df) + 2
            startrow3 = startrow2 + len(duplicates_matrix_df) + 2 if duplicates_matrix_df is not None else startrow2 + 2
            startrow4 = startrow3 + len(duplicated_rows_df) + 2
            
            # Write duplicates matrix if it exists and is not empty
            if duplicates_matrix_df is not None and not duplicates_matrix_df.empty:
                duplicates_matrix_df.to_excel(writer, sheet_name='Summary', 
                                            index=False, startrow=startrow2)
                duplicated_rows_df.to_excel(writer, sheet_name='Summary', 
                                        index=False, startrow=startrow3)
            
            # Write risk summary
            risk_summary_df.to_excel(writer, sheet_name='Summary', 
                                   index=True, startrow=startrow4)
        
        # Write missing participants to separate sheet
        missing_matrix_df.to_excel(writer, sheet_name='Missing Participants', index=False)
    
    output.seek(0)
    return output


def create_summary_excel_with_download(is_main_study=False, is_booster=False, 
                                     baseline_data=None, three_month_data=None,
                                     booster_data=None, download_label="Download Summary Report"):
    
    if is_main_study and baseline_data and three_month_data:
        # Create ZIP file with both baseline and 3-month reports
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Create baseline Excel file
            baseline_excel = create_summary_excel(
                baseline_data['summary_df'],
                baseline_data['duplicates_matrix_df'],
                baseline_data['duplicated_rows_df'],
                baseline_data['risk_summary_df'],
                baseline_data['missing_matrix_df'],
                is_booster=False
            )
            zip_file.writestr("Baseline_summary_report.xlsx", baseline_excel.getvalue())
            
            # Create 3-month Excel file
            three_month_excel = create_summary_excel(
                three_month_data['summary_df'],
                three_month_data['duplicates_matrix_df'],
                three_month_data['duplicated_rows_df'],
                three_month_data['risk_summary_df'],
                three_month_data['missing_matrix_df'],
                is_booster=False
            )
            zip_file.writestr("3Month_summary_report.xlsx", three_month_excel.getvalue())
        
        zip_buffer.seek(0)
        
        # Show single download button for ZIP file
        st.download_button(
            download_label,
            data=zip_buffer,
            file_name="Main_Study_summary_reports.zip",
            mime="application/zip"
        )
        
    elif is_booster and booster_data:
        # Single Excel file for booster data
        output = create_summary_excel(
            booster_data['summary_df'],
            booster_data['duplicates_matrix_df'],
            booster_data['duplicated_rows_df'],
            booster_data['risk_summary_df'],
            booster_data['missing_matrix_df'],
            is_booster=True
        )
        
        # Show download button for single Excel file
        st.download_button(
            download_label,
            data=output,
            file_name="Booster_summary_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    else:
        st.error("Please provide the appropriate data for the selected study type.")

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

st.write("You can also choose to upload booster files for additional data processing. If you are working with the main study, please ensure you upload exactly 2 questionnaire and 2 gait files for baseline and 3-month follow-up.")
is_booster = st.checkbox("Booster Data", value=True, key="booster")
is_main_study = st.checkbox("Main Study Data", value=False, key="main_study")

if not (use_ZM or use_QN or use_age_fall_history):
    st.warning("Please select at least one data type to proceed.")
    st.stop()

df_meta = pd.read_excel("/1TB/wYr_model/wYr_thresholds.xlsx", sheet_name="Thresholds")

gait_keywords = None
questionnaire_keywords = None
age_fall_history_keywords = None

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
ZM_files = st.file_uploader("Upload your Gait data file here", type=["csv", "xlsx"], accept_multiple_files=True, key="zm_file_uploader") if use_ZM else None
QN_files = st.file_uploader("Upload your Questionnaire data file here", type=["csv", "xlsx"], accept_multiple_files=True, key = "qn_file_uploader") if use_QN or use_age_fall_history else None

cleaned_files = {}

if is_booster:
    # st.write("Debug: ZM_files =", ZM_files)
    # st.write("Debug: QN_files =", QN_files)
    if (not ZM_files or len(ZM_files) == 0) and (not QN_files or len(QN_files) == 0):
        st.warning("Please upload at least one data file to proceed.")
        st.stop()
    summary_df, duplicates_matrix_df, missing_matrix_df, duplicated_rows_df, zm_Combined_file, qn_Combined_file = combine_uploaded_booster_files(ZM_files, QN_files, use_QN, use_age_fall_history)

    cleaned_files = {
        "ZM" : zm_Combined_file,
        "Questionnaire": qn_Combined_file
    }

    target = None
    try:
        target = f_generate_TARGET_dataset(cleaned_files, use_ZM=use_ZM, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    df_final = process_risk_score(target, df_meta, parameters, means, stddevs, wts, dirns, 
                                   use_QN=use_QN, use_ZM=use_ZM, use_age_fall_history=use_age_fall_history)
    
    risk_summary_df = create_risk_summary(df_final, prediction_col='prediction')

    booster_data = {
        'summary_df': summary_df,
        'duplicates_matrix_df': duplicates_matrix_df,
        'missing_matrix_df': missing_matrix_df,
        'duplicated_rows_df': duplicated_rows_df,
        'risk_summary_df': risk_summary_df
    }

    create_summary_excel_with_download(is_booster=True, booster_data=booster_data)
    processed_csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed Data", data=processed_csv, file_name="Booster_processed_data.csv", mime="text/csv")
     

if is_main_study:
    if len(QN_files) != 2 or len(ZM_files) != 2:
        st.warning("Please upload exactly 2 questionnaire and 2 gait files for the main study (baseline & 3month).")
        st.stop()

    data_dict = combine_main_study_files(QN_files, ZM_files)
    cleaned_files_baseline = {
        "ZM": data_dict['cleaned_data']['zm_baseline'],
        "Questionnaire": data_dict['cleaned_data']['qn_baseline']
    }
    cleaned_files_3month = {
        "ZM": data_dict['cleaned_data']['zm_3month'],
        "Questionnaire": data_dict['cleaned_data']['qn_3month']
    }

    target_baseline = None
    target_3month = None
    try:
        target_baseline = f_generate_TARGET_dataset(cleaned_files_baseline, use_ZM=use_ZM, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
        target_3month = f_generate_TARGET_dataset(cleaned_files_3month, use_ZM=use_ZM, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    df_final_baseline = process_risk_score(target_baseline, df_meta, parameters, means, stddevs, wts, dirns,
                                           use_QN=use_QN, use_ZM=use_ZM, use_age_fall_history=use_age_fall_history)
    df_final_3month = process_risk_score(target_3month, df_meta, parameters, means, stddevs, wts, dirns,
                                           use_QN=use_QN, use_ZM=use_ZM, use_age_fall_history=use_age_fall_history)
    
    risk_summary_df_baseline = create_risk_summary(df_final_baseline, prediction_col='prediction')
    risk_summary_df_3month = create_risk_summary(df_final_3month, prediction_col='prediction')

    data_dict['diagnostics']['baseline']['risk_summary_df'] = risk_summary_df_baseline
    data_dict['diagnostics']['3month']['risk_summary_df'] = risk_summary_df_3month

    create_summary_excel_with_download(is_main_study=True,
                                       baseline_data=data_dict['diagnostics']['baseline'],
                                       three_month_data=data_dict['diagnostics']['3month'])
    
    processed_csv_baseline = df_final_baseline.to_csv(index=False).encode('utf-8')
    processed_csv_3month = df_final_3month.to_csv(index=False).encode('utf-8')

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("Baseline_processed_data.csv", processed_csv_baseline)
        zip_file.writestr("3Month_processed_data.csv", processed_csv_3month)
    zip_buffer.seek(0)
    st.download_button("Download Processed Data", data=zip_buffer, file_name="Main_Study_processed_data.zip", mime="application/zip")

if not (is_booster or is_main_study):
    ZM_cleaned_files = []
    QN_cleaned_files = []
    if ZM_files:
        for files in ZM_files:
            df = read_zm_file(files)
            ZM_cleaned_files.append(df)

    if QN_files:
        for files in QN_files:
            df = read_qn_file(files, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
            QN_cleaned_files.append(df)

    cleaned_files = {
        "ZM": pd.concat(ZM_cleaned_files, ignore_index=True).drop_duplicates(subset='Participant', keep='first') 
            if ZM_cleaned_files else pd.DataFrame(),

        "Questionnaire": pd.concat(QN_cleaned_files, ignore_index=True).drop_duplicates(subset='Participant', keep='first') 
                        if QN_cleaned_files else pd.DataFrame()
    }
    
    target = f_generate_TARGET_dataset(cleaned_files, use_ZM=use_ZM, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
    if target is None:
        st.warning("No valid data found. Please upload the selected data files to proceed.")
        st.stop()
    df_final = process_risk_score(target, df_meta, parameters, means, stddevs, wts, dirns, 
                                   use_QN=use_QN, use_ZM=use_ZM, use_age_fall_history=use_age_fall_history)
    processed_csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed Data", data=processed_csv, file_name="Processed_data.csv", mime="text/csv")
    

st.header("Processing and Results")
st.write(target)
st.write("Once uploaded, the system will process the data and calculate a risk score and provide predictions based on the provided data.")
st.write("You can then download the updated file for further review or analysis.")

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






