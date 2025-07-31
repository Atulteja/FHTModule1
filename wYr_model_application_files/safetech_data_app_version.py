import pandas as pd
import streamlit as st

from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

def generate_diagnostics(df1, df2, label1="DF1", label2="DF2", id_col="Participant"):
    # Drop duplicates
    df1_nodup = df1.drop_duplicates(subset=id_col, keep='first')
    df2_nodup = df2.drop_duplicates(subset=id_col, keep='first')

    # Summary of rows and duplicates
    summary_df = pd.DataFrame({
        "number of rows_raw": [len(df1), len(df2), len(df1_nodup)],
        "number of duplicates": [
            df1.duplicated(subset=id_col, keep=False).sum(),
            df2.duplicated(subset=id_col, keep=False).sum(),
            0
        ],
        "number of rows_after removing duplicates": [len(df1_nodup), len(df2_nodup), len(df1_nodup)]
    }, index=[label1, label2, "Processed"])

    # Duplicates Matrix
    dups1 = df1[df1.duplicated(subset=id_col, keep=False)][id_col].unique()
    dups2 = df2[df2.duplicated(subset=id_col, keep=False)][id_col].unique()
    max_len = max(len(dups1), len(dups2))
    dups_matrix = pd.DataFrame({
        f"Duplicates in {label1}": list(dups1) + [None] * (max_len - len(dups1)),
        f"Duplicates in {label2}": list(dups2) + [None] * (max_len - len(dups2))
    })

    # Missing IDs Matrix
    ids1 = set(df1_nodup[id_col])
    ids2 = set(df2_nodup[id_col])
    in_2_not_1 = sorted(ids2 - ids1)
    in_1_not_2 = sorted(ids1 - ids2)
    max_missing = max(len(in_2_not_1), len(in_1_not_2))
    missing_matrix = pd.DataFrame({
        f"In {label2} Missing in {label1}": in_2_not_1 + [None] * (max_missing - len(in_2_not_1)),
        f"In {label1} Missing in {label2}": in_1_not_2 + [None] * (max_missing - len(in_1_not_2)),
    })

    # Duplicated Rows with source tag
    dup1_rows = df1[df1.duplicated(subset=id_col, keep=False)].copy()
    dup1_rows["Dataset"] = label1
    dup2_rows = df2[df2.duplicated(subset=id_col, keep=False)].copy()
    dup2_rows["Dataset"] = label2
    duplicated_rows_df = pd.concat([dup1_rows, dup2_rows], ignore_index=True)

    return summary_df, dups_matrix, missing_matrix, duplicated_rows_df

def read_zm_file(file):
    zm = ZURICHMOVEDATA(file)
    zm.read_dataset()
    return zm.dataset

def read_qn_file(file, use_QN=True, use_age_fall_history=False):
    if use_QN and use_age_fall_history:
        features = ['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
    elif use_QN and not use_age_fall_history:
        features = ['ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
    elif use_age_fall_history and not use_QN:
        features = ['Fall_2', 'Age']
    else:
        features = None
    qn = QUESTIONNAIREDATA(file, features=features)
    qn.read_dataset()
    return qn.dataset

def combine_uploaded_booster_files(zm_files, qn_files, use_QN, use_age_fall_history):
    zm_dfs = []
    for file in zm_files:
        df = read_zm_file(file)
        zm_dfs.append(df)

    # Process QN files
    qn_dfs = []
    for file in qn_files:
        df = read_qn_file(file, use_QN=use_QN, use_age_fall_history=use_age_fall_history)
        st.write(f"columns in QN file {file.name}:", df.columns.tolist())
        qn_dfs.append(df)
    
    # Combine datasets
    zm_combined = pd.concat(zm_dfs, ignore_index=True)
    qn_combined = pd.concat(qn_dfs, ignore_index=True)
    zm_combined_nodup = zm_combined.drop_duplicates(subset='Participant', keep='first')
    qn_combined_nodup = qn_combined.drop_duplicates(subset='Participant', keep='first')


    summary_df, duplicates_matrix_df, missing_matrix_df, duplicated_rows_df = generate_diagnostics(zm_combined, qn_combined, label1="ZM", label2="QN", id_col="Participant")
    return summary_df, duplicates_matrix_df, missing_matrix_df, duplicated_rows_df, zm_combined_nodup, qn_combined_nodup

def combine_main_study_files(qn_files, zm_files):
    
    qn_baseline_df = read_qn_file(qn_files[0], use_QN=True, use_age_fall_history=True)
    qn_3month_df = read_qn_file(qn_files[1], use_QN=True, use_age_fall_history=True)
    zm_baseline_df = read_zm_file(zm_files[0])
    zm_3month_df = read_zm_file(zm_files[1])

    qn_3month_df = qn_3month_df.drop(columns=['Age', 'Fall_2'], errors='ignore')
    qn_3month_df = qn_3month_df.merge(
        qn_baseline_df[['Participant', 'Age', 'Fall_2']],
        on='Participant',
        how='left'
    )

    summary_baseline_df, duplicates_baseline_matrix_df, missing_baseline_matrix_df, duplicated_baseline_rows_df = generate_diagnostics(zm_baseline_df, qn_baseline_df, label1="ZM", label2="QN", id_col="Participant")

    summary_3month_df, duplicates_3month_matrix_df, missing_3month_matrix_df, duplicated_3month_rows_df = generate_diagnostics(zm_3month_df, qn_3month_df, label1="ZM", label2="QN", id_col="Participant")

    qn_baseline_df_cleaned = qn_baseline_df.drop_duplicates(subset='Participant', keep='first')
    qn_3month_df_cleaned = qn_3month_df.drop_duplicates(subset='Participant', keep='first')
    zm_baseline_df_cleaned = zm_baseline_df.drop_duplicates(subset='Participant', keep='first')
    zm_3month_df_cleaned = zm_3month_df.drop_duplicates(subset='Participant', keep='first') 

    return {
        'cleaned_data': {
            'qn_baseline': qn_baseline_df_cleaned,
            'qn_3month': qn_3month_df_cleaned,
            'zm_baseline': zm_baseline_df_cleaned,
            'zm_3month': zm_3month_df_cleaned
        },
        'diagnostics': {
            'baseline': {
                'summary': summary_baseline_df,
                'duplicates_matrix': duplicates_baseline_matrix_df,
                'missing_matrix': missing_baseline_matrix_df,
                'duplicated_rows': duplicated_baseline_rows_df
            },
            '3month': {
                'summary': summary_3month_df,
                'duplicates_matrix': duplicates_3month_matrix_df,
                'missing_matrix': missing_3month_matrix_df,
                'duplicated_rows': duplicated_3month_rows_df
            }
        }
    }



