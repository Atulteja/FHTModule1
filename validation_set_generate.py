import pandas as pd
from generate_dataset import PROSPECTIVEDATA, QUESTIONNAIREDATA, ZURICHMOVEDATA, DATASET

def generate_train_test_dataset(paths, label):
    
    ids = pd.read_excel(paths['split_ids'], usecols=[f"Case_num_{label}"])
    ids = ids[f"Case_num_{label}"].tolist()

    zm = ZURICHMOVEDATA(paths['ZM'])
    zm.read_dataset()
    df_zm = zm.dataset 

    questionnaire = QUESTIONNAIREDATA(
        paths['Questionnaires'],
        features=['Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted']
    )
    questionnaire.read_dataset()
    df_questionnaire = questionnaire.dataset

    df_followup = pd.read_excel(paths['Prospective'])

    df_zm_filtered = df_zm[df_zm["Participant"].isin(ids)]
    df_questionnaire_filtered = df_questionnaire[df_questionnaire["Participant"].isin(ids)]
    df_followup_filtered = df_followup[df_followup["Case_num"].isin(ids)]

    df_zm_filtered.to_excel(f"{label}ing_ZM.xlsx", index=False)
    df_questionnaire_filtered.to_excel(f"{label}ing_questionnaire.xlsx", index=False)
    df_followup_filtered.to_excel(f"{label}ing_followup.xlsx", index=False)

    print(f"Training/Test datasets for `{label}` set saved.")

    return df_zm_filtered, df_questionnaire_filtered, df_followup_filtered




if __name__ == "__main__":
    paths = {
        'ZM': "//1TB/Dataset_Feb2025/TARGETZMParameters_All_20241015.xlsx",
        'Questionnaires': "/1TB/Dataset_Feb2025/TARGET 5 November 2024 dataset to SEC_051124 numeric n=2291.xlsx",
        'Prospective': "/1TB/Dataset_Feb2025/TARGET follow-up 18.02.2025 to SEC 26022025 numeric.xls",
        'split_ids': "/1TB/wYr_model/dataset_split_ids.xlsx"
    }

    generate_train_test_dataset(paths, label="train")
    generate_train_test_dataset(paths, label="tune")
    generate_train_test_dataset(paths, label="test")