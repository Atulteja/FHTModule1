import numpy as np
import pandas as pd


class ZURICHMOVEDATA:
    def __init__(self, path):
        self.dataset = None
        self.path = path

    def read_dataset(self):
        self.dataset = pd.read_excel(self.path)


class QUESTIONNAIREDATA:
    def __init__(self, path, features=None) -> None:
        self.features = features
        self.dataset = None
        self.path = path

    def read_dataset(self):
        self.dataset = pd.read_excel(self.path)
        self.clean_dataset()

    def clean_dataset(self):
        self.dataset = self.dataset.replace({'Fall_2': {2: 0, 1: 1, 999: 0}})
        self.dataset = self.dataset.replace({'Age': {888: 65, 999: 65}})
        self.dataset = self.dataset.replace({'ICONFES_Score_Adjusted': {888: 0}})
        self.dataset = self.dataset.replace({'IPAQ_Cat': {888: 0}})
        self.dataset = self.dataset.replace({'MOCA_Score_Adjusted': {888: 21}})
        # 'Fall_2', 'Age', 'ICONFES_Score_Adjusted', 'IPAQ_Cat', 'MOCA_Score_Adjusted'


        self.dataset.rename({'Case_num': 'Participant'}, inplace=True, axis=1)
        self.features[:0] = ['Participant']
        if self.features is not None:
            self.dataset = self.dataset[self.features]


class DATASET():

    def __init__(self) -> None:
        self.dataset = None

    def merge_datasets(self, df1, df2, merge_columns):
        self.dataset = df1.merge(df2, left_on=merge_columns['first_dataset'], right_on=merge_columns['second_dataset'])


class PROSPECTIVEDATA:

    def __init__(self, path) -> None:
        self.dataset = None
        self.path = path

    def read_dataset(self) -> object:
        self.dataset = pd.read_excel(self.path)

    def generate_labels(self, num_followups=1):
        # df_dummy = self.dataset[['Case_num', 'Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a', 'Q5_S2a', 'Q6_S2a']]
        df_dummy = self.dataset[['Case_num', 'Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a']]
        df_dummy = df_dummy.replace({
            'Not applicable': np.nan,
            'Withdrawal': np.nan,
            'Passed away': np.nan,
            'Refused': np.nan,
            'Uncontactable': np.nan,
            "Don't know": np.nan,
            "No-Temporarily unavailable": np.nan,
            "No-Reason other than temporarily unavailable": np.nan,
            888 : np.nan,
            777: np.nan,
            333: np.nan,
            444 : np.nan,
            2: np.nan,
            3: np.nan,
            'Yes': 1,
            'No': 0,
            'Follow-up not due': 666
        })
        df_label = pd.DataFrame([], columns=['Participant', 'num_followups', 'label'])
        for participant in df_dummy['Case_num']:
            lst = ['', '', '']
            lst[0] = int(participant)
            # if 1 in df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a', 'Q5_S2a', 'Q6_S2a']].values:
            if 1 in df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a']].values:
                lst[2] = 1
            else:
                val = (df_dummy[df_dummy['Case_num'] == participant]['Q1_S2a']
                       * df_dummy[df_dummy['Case_num'] == participant]['Q2_S2a']
                       * df_dummy[df_dummy['Case_num'] == participant]['Q3_S2a']
                       * df_dummy[df_dummy['Case_num'] == participant]['Q4_S2a']).values[0]
                
                # val = (df_dummy[df_dummy['Case_num'] == participant]['Q1_S2a']
                #        * df_dummy[df_dummy['Case_num'] == participant]['Q2_S2a']
                #        * df_dummy[df_dummy['Case_num'] == participant]['Q3_S2a']
                #        * df_dummy[df_dummy['Case_num'] == participant]['Q4_S2a']
                #        * df_dummy[df_dummy['Case_num'] == participant]['Q5_S2a']
                #        * df_dummy[df_dummy['Case_num'] == participant]['Q6_S2a']).values[0]

                lst[2] = val
            
            # lst[2] = df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a']].values[0].tolist().count(0)
            # +  df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a']].values[0].tolist().count(1)
                
            # lst[1] = 6 - df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a', 'Q5_S2a', 'Q6_S2a']].values[0].tolist().count(666)
            lst[1] = 4 - df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a']].values[0].tolist().count(666)
            
            df_label.loc[len(df_label)] = lst
        
        self.labels = df_label[df_label['num_followups']>=num_followups]
        # pass
# df_dummy[df_dummy['Case_num'] == participant][['Q1_S2a', 'Q2_S2a', 'Q3_S2a', 'Q4_S2a']].dropna(axis=1).shape

if __name__ == '__main__':
    paths = {"ZurichMOVE_data": "/1TB/TARGET_analysis/Threshold_based_modelling_v3/Data/TARGETZMParameters.xlsx",
             "Questionnaire_data": "/1TB/TARGET_analysis/Threshold_based_modelling_v3/Data/TARGET B2 06 Nov 23 dataset to SEC_041123 n=642.xlsx",
             "Prospective_data": "/1TB/TARGET_analysis/Threshold_based_modelling_v3/Data/TARGET_follow_up_B2_31.12.2023_to_SEC_020224.xlsx"}

    zm = ZURICHMOVEDATA(paths['ZurichMOVE_data'])
    zm.read_dataset()

    questionnaire = QUESTIONNAIREDATA(paths['Questionnaire_data'], features=['Fall_2'])
    questionnaire.read_dataset()

    followup = PROSPECTIVEDATA(paths['Prospective_data'])
    followup.read_dataset()
    followup.generate_labels()

    target = DATASET()
    target.merge_datasets(zm.dataset, questionnaire.dataset,
                          {'first_dataset': 'Participant', 'second_dataset': 'Participant'})

    complete_dataset = DATASET()
    complete_dataset.merge_datasets(target.dataset, followup.labels,
                          {'first_dataset': 'Participant', 'second_dataset': 'Participant'})

    print('completed')
