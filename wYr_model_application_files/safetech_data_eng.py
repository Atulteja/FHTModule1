import pandas as pd

from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

def combine_booster_files(path):
    zm1= ZURICHMOVEDATA(path['Booster1ZM'])
    zm2= ZURICHMOVEDATA(path['Booster2ZM'])
    zm3= ZURICHMOVEDATA(path['Booster3ZM'])
    zm1.read_dataset()
    zm2.read_dataset()
    zm3.read_dataset()

    zm_df = pd.concat([zm1.dataset, zm2.dataset, zm3.dataset], ignore_index=True)
    zm_duplicates = zm_df[zm_df.duplicated(subset='Participant', keep=False)]['Participant'].unique()
    if len(zm_duplicates) > 0:
        print(f"Duplicated Participant IDs in ZM before dropping: {sorted(zm_duplicates)}")
    else:
        print("No duplicates found in ZM dataset.")
    zm_df = zm_df.drop_duplicates(subset='Participant', keep='first')

    qn1 = QUESTIONNAIREDATA(path['Booster1QN'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat'])
    qn2 = QUESTIONNAIREDATA(path['Booster2QN'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat'])
    qn3 = QUESTIONNAIREDATA(path['Booster3QN'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat'])
    qn1.read_dataset()
    qn2.read_dataset()
    qn3.read_dataset()

    qn_df = pd.concat([qn1.dataset, qn2.dataset, qn3.dataset], ignore_index=True)
    qn_duplicates = qn_df[qn_df.duplicated(subset='Participant', keep=False)]['Participant'].unique()
    if len(qn_duplicates) > 0:
        print(f"Duplicated Participant IDs in QN before dropping: {sorted(qn_duplicates)}")
    else:
        print("No duplicates found in QN dataset.")
    qn_df = qn_df.drop_duplicates(subset='Participant', keep='first')

    proc_df = pd.read_csv(path['BoosterProcessed'])

    #zm_df.to_excel('booster_combined_zm.xlsx', index=False)
    #qn_df.to_excel('booster_combined_qn.xlsx', index=False)

    def report_missing(zm_df, qn_df, proc_df, label):
        zm_participants = set(zm_df['Participant'])
        qn_participants = set(qn_df['Participant'])
        proc_participants = set(proc_df['Participant'])

        missing_in_qn = zm_participants - qn_participants
        missing_in_zm = qn_participants - zm_participants
        missing_in_proc_qn = qn_participants - proc_participants
        missing_in_proc_zm = zm_participants - proc_participants

        # if missing_in_zm:
        #     print(f"participants in {label} qn but not in zm : {sorted(missing_in_zm)}")
        # else:
        #     print(f"no missing in qn for {label}")

        if missing_in_proc_qn:
            print(f"missing in proccessed but in qn: {sorted(missing_in_proc_qn)}")
        else: 
            print(f"no missing in processed from qn")

        if missing_in_proc_zm:
            print(f"missing in proccessed but in zm: {sorted(missing_in_proc_zm)}")
        else: 
            print(f"no missing in processed from zm")

    
    # report_missing(zm_df, qn_df, proc_df, "combined datasets")

    return zm_df, qn_df

def main_study(path):

    qn1 = QUESTIONNAIREDATA(paths['MainBaselineQN'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted'])
    qn2 = QUESTIONNAIREDATA(paths['Main3monthQN'], features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted'])
    zm1= ZURICHMOVEDATA(paths['MainBaselineZM'])
    zm2= ZURICHMOVEDATA(paths['Main3monthZM'])

    qn1.read_dataset()
    qn2.read_dataset()
    zm1.read_dataset()
    zm2.read_dataset()

    qn1_df = qn1.dataset
    qn2_df = qn2.dataset
    zm1_df = zm1.dataset
    zm2_df = zm2.dataset

    qn2_df = qn2_df.drop(columns=['Age', 'Fall_2'], errors='ignore')

    qn2_df = qn2_df.merge(
        qn1_df[['Participant', 'Age', 'Fall_2']],
        on='Participant',
        how='left'
    )
    
    qn2_df.to_excel("3month_updated.xlsx", index=False)

    participants_qn1 = set(qn1_df['Participant'])
    participants_qn2 = set(qn2_df['Participant'])
    participants_zm1 = set(zm1_df['Participant'])
    participants_zm2 = set(zm2_df['Participant'])


    missqn2 = participants_qn1 - participants_qn2
    missqn1 = participants_qn2 - participants_qn1
    misszm2 = participants_zm1 - participants_zm2
    misszm1 = participants_zm2 - participants_zm1

    missbaselinezm = participants_qn1 - participants_zm1
    missbaselineqn = participants_zm1 - participants_qn1
    miss3monthzm = participants_qn2 - participants_zm2
    miss3monthqn = participants_zm2 - participants_qn2

    # if missbaselinezm:
    #     print(F"participants in baseline but missing in 3 month: {misszm2}")
    # else: 
    #     print("all of baseline in 3 month")

    # if misszm2:
    #     print(F"participants in 3 month but missing in baseline: {misszm1}")
    # else: 
    #     print("all of 3month in baseline")

    # if missbaselinezm:
    #     print(f"participants in baseline qn but not in baseline zm: {missbaselinezm}")
    # else:
    #     print("No participants missing from Baseline ZM (all Baseline QN participants found)")
    
    # if missbaselineqn:
    #     print(f"participants in baseline zm but not in baseline qn: {missbaselineqn}")
    # else:
    #     print("No participants missing from Baseline qn (all Baseline zm participants found)")

    # if miss3monthzm:
    #     print(f"participants in 3 month qn but not in zm: {miss3monthzm}")
    # else:
    #     print("no participant missing from 3 month zm")
    
    # if miss3monthqn:
    #     print(f"participants in 3 month zm but not in qn: {miss3monthqn}")
    # else:
    #     print("no participant missing from 3 month qn")
    

    #print("Updated second dataset saved as 'second_dataset_updated.xlsx'")


if __name__ == "__main__":

    paths = {}
    paths['Booster1ZM'] = "/1TB/SAFETECH/Booster/02_ProcessedData/booster1/SAFETECHZMParameters_Booster1_20250531.xlsx"
    paths['Booster2ZM'] = "/1TB/SAFETECH/Booster/02_ProcessedData/booster2/SAFETECHZMPParameters_Booster2_20250701.xlsx"
    paths['Booster3ZM'] = "/1TB/SAFETECH/Booster/02_ProcessedData/booster3/SAFETECHZMPParameters_Booster3_20250704.xlsx"
    paths['Booster1QN'] = "/1TB/SAFETECH/Booster/03_Questionnaire/Batch 1 (128)/Fall risk score variable_Batch 1 (128).xlsx"
    paths['Booster2QN'] = "/1TB/SAFETECH/Booster/03_Questionnaire/Batch 2 (147)/Fall risk score variable_Batch 2 (147).xlsx"
    paths['Booster3QN'] = "/1TB/SAFETECH/Booster/03_Questionnaire/Batch 3 (90)/Fall risk score variable_Batch 3 (90).xlsx"
    paths['BoosterProcessed'] = "/1TB/SAFETECH/Booster/04_RiskScores/booster_riskscore_predictions.csv"
    paths['MainBaselineQN'] = "/1TB/SAFETECH/Main_study/03_Questionnaire/Baseline (Class 1-10)/Baseline_Fall risk variable (Class 1-10).xlsx"
    paths['Main3monthQN'] = "/1TB/SAFETECH/Main_study/03_Questionnaire/3rd-month (Class1-10)/3rd-month_Fall risk variable (Class 1-10).xlsx"
    paths['MainBaselineZM'] = "/1TB/SAFETECH/Main_study/02_ZMParameters/SAFETECZMParameters_baseline_v1_20250621.xlsx"
    paths['Main3monthZM'] = "/1TB/SAFETECH/Main_study/02_ZMParameters/SAFETECZMParameters_thirdmonth_v1_20250621.xlsx"

    # paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"
    # paths['ZM'] = "/1TB/Dataset_Feb2025/Training_ZM.xlsx"
    # paths['Questionnaires'] = "/1TB/Dataset_Feb2025/Training_questionnaire.xlsx"
    # paths['Prospective'] = "/1TB/Dataset_Feb2025/Training_followup.xlsx"
    # paths["Model_information"] = "/1TB/wYr_model/wYr_thresholds.xlsx"

    combine_booster_files(paths)
    # main_study(paths)

