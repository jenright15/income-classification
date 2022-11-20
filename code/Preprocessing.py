'''
Preprocesses the data for data modeling.

@author: John Enright
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing

class IncomePreprocess:
    '''
    Preprocesses the data for data modeling. 
    Cleans, labels, and encodes features within the dataset.
    '''
    def __init__(self):
        return None


    def label_features(self, train_set, test_set):
        '''
        Label features in the dataset.
        '''

        column_names={
            0 : 'age',
            1 : 'class_of_work',
            2 : 'industry_code',
            3 : 'occupation_code',
            4 : 'education',
            5 : 'wage_per_hour',
            6 : 'enrolled_in_edu_inst_last_wk',
            7 : 'marital_status',
            8 : 'major_industry_code',
            9: 'major_occupation_code',
            10: 'race',
            11: 'hispanic_origin',
            12: 'sex',
            13: 'member_of_labor_union',
            14: 'reason_for_unemployment',
            15: 'full_or_part_time_employment_stat',
            16: 'capital_gains',
            17: 'capital_losses',
            18: 'dividends_from_stocks',
            19: 'tax_filer_status',
            20: 'region of previous residence',
            21: 'state_of_previous_residence',
            22: 'detailed_household_and_family_stat',
            23: 'detailed_household_summary_in_household',
            24: 'instance_weight',
            25: 'migration_code_change_in_msa',
            26: 'migration_code_change_in_reg',
            27: 'migration_code_move_within_reg',
            28: 'live_in_this_house_1_year_ago',
            29: 'migration_prev_res_in_sunbelt',
            30: 'num_persons_worked_for_employer',
            31: 'family_member_under_18',
            32: 'country_of_birth_father',
            33: 'country_of_birth_mother',
            34: 'country_of_birth_self',
            35: 'citizenship',
            36: 'own_business_or_self_employed',
            37: 'fill_inc_questionnaire_for_veterans_admin',
            38: 'veteran_benefits',
            39: 'weeks_worked_in_year',
            40: 'year',
            41: 'y'    
    }

        labeled_train = train_set.rename(columns= column_names)
        labeled_test = test_set.rename(columns= column_names)

        return labeled_train, labeled_test
    

    
    def categorize_features(self, columns, train_set, test_set):
        '''
        Categorizes the features in the dataset using sklearn labelencoder.
        '''

        encoder = preprocessing.LabelEncoder()
        encoded_train = train_set.copy()
        encoded_test = test_set.copy()

        for column in columns:
            encoded_train[column] = encoder.fit_transform(encoded_train[column])
            encoded_test[column] = encoder.fit_transform(encoded_test[column])
        
        return encoded_train, encoded_test
    
    # Categorize household information
    def categorize_household_info(self, df):
        '''
        Categorizes household information.
        '''
        df_copy = df.copy()
        mp = {
            ' Householder': 1,
            ' Spouse of householder': 2,
            ' Child <18 never marr RP of subfamily': 3,
            ' Child <18 never marr not in subfamily': 4,
            ' Child <18 ever marr RP of subfamily': 5,
            ' Child <18 spouse of subfamily RP': 6,
            ' Child <18 ever marr not in subfamily': 7,
            ' Child 18+ ever marr RP of subfamily': 8,
            ' Child 18+ never marr Not in a subfamily': 9,
            ' Child 18+ never marr RP of subfamily' : 10,
            ' Child 18+ spouse of subfamily RP': 11,
            ' Child 18+ ever marr Not in a subfamily': 12,
            ' Grandchild <18 never marr RP of subfamily': 23,
            ' Grandchild <18 never marr child of subfamily RP':24,
            ' Grandchild <18 never marr not in subfamily': 25,
            ' Grandchild <18 ever marr RP of subfamily': 26,
            ' Grandchild <18 ever marr not in subfamily': 29,
            ' Grandchild 18+ never marr RP of subfamily': 30,
            ' Grandchild 18+ never marr not in subfamily': 31,   
            ' Grandchild 18+ ever marr RP of subfamily': 32,   
            ' Grandchild 18+ spouse of subfamily RP': 33,
            ' Grandchild 18+ ever marr not in subfamily': 34,
            ' Other Rel <18 never married RP of subfamily': 35,
            ' Other Rel <18 never marr child of subfamily RP': 36, 
            ' Other Rel <18 never marr not in subfamily': 37,
            ' Other Rel <18 ever marr RP of subfamily': 38,
            ' Other Rel <18 spouse of subfamily RP': 39,
            ' Other Rel <18 ever marr not in subfamily': 40,
            ' Other Rel 18+ never marr RP of subfamily': 41,
            ' Other Rel 18+ never marr not in subfamily': 42,
            ' Other Rel 18+ ever marr RP of subfamily': 43,
            ' Other Rel 18+ spouse of subfamily RP': 44,
            ' Other Rel 18+ ever marr not in subfamily': 45,
            ' RP of unrelated subfamily': 46,
            ' Spouse of RP of unrelated subfamily': 47,
            ' Child under 18 of RP of unrel subfamily': 48,
            ' Nonfamily householder': 49,
            ' Secondary individual': 50,
            ' In group quarters': 51
        }

        df_copy['detailed_household_and_family_stat'] = df_copy['detailed_household_and_family_stat'].map(mp)
        return df_copy

    def categorize_education(self, df):
        '''
        Categorizes education information.
        '''
        df_copy = df.copy()
        mp={
            ' Children': 0,
            ' Less than 1st grade': 1,
            ' 1st 2nd 3rd or 4th grade': 2,
            ' 5th or 6th grade':3,
            ' 7th and 8th grade':4,
            ' 9th grade': 5,
            ' 10th grade': 6,
            ' 11th grade': 7,
            ' 12th grade no diploma': 8,
            ' High school graduate' : 9,
            ' Some college but no degree': 10,
            ' Associates degree-occup /vocational': 11,
            ' Associates degree-academic program': 12,
            ' Bachelors degree(BA AB BS)':13,
            ' Masters degree(MA MS MEng MEd MSW MBA)': 14,
            ' Prof school degree (MD DDS DVM LLB JD)': 15,
            ' Doctorate degree(PhD EdD)': 16    
            }
        df_copy['education'] = df_copy['education'].map(mp)
        return df_copy
    
    def categorize_work_class(self, df):
        '''
        Categorizes work class information.
        '''
        df_copy = df.copy()
        mp={
            ' Not in universe' : 0,
            ' Private': 1,
            ' Federal government': 2,
            ' State government': 2,
            ' Local government': 2,
            ' Self-employed-incorporated': 3,
            ' Self-employed-not incorporated': 3,
            ' Without pay': 4,
            ' Never worked': 4
            }
        df_copy['class_of_work'] = df_copy['class_of_work'].map(mp)
        return df_copy
    
    def categorize_marital(self, df):
        '''
        Categorizes marital information.
        '''
        df_copy = df.copy()
        mp={
            ' Married-civilian spouse present' : 1,
            ' Married-A F spouse present': 2,
            ' Married-spouse absent': 3,
            ' Widowed': 4,
            ' Divorced': 5,
            ' Separated': 6,
            ' Never married':7
        }
        
        df_copy['marital_status'] = df_copy['marital_status'].map(mp)
        return df_copy

    def categorize_financial_info(self, df):
        '''
        Categorizes financial information.
        '''
        df_copy = df.copy()
        df_copy.loc[df_copy['capital_gains'] >0, 'has_gains']= 1
        df_copy.loc[df_copy['capital_gains'] == 0, 'has_gains']= 0
        
        df_copy.loc[df_copy['capital_losses'] >0, 'has_losses']= 1
        df_copy.loc[df_copy['capital_losses'] == 0, 'has_losses']= 0
        
        df_copy.loc[df_copy['dividends_from_stocks'] >0, 'has_stock']= 1
        df_copy.loc[df_copy['dividends_from_stocks'] == 0, 'has_stock']= 0

        return df_copy
    
    def categorize_household_summary(self, df):
        '''
        Categorizes person information.
        '''
        df_copy = df.copy()
        mp = {
            ' Householder': 1,
            ' Spouse of householder': 2,
            ' Child under 18 never married': 3,
            ' Child under 18 ever married': 4,
            ' Child 18 or older': 5,
            ' Other relative of householder' : 6,
            ' Nonrelative of householder' : 7,
            ' Group Quarters- Secondary individual' : 8
            }
        df_copy['detailed_household_summary_in_household'] = df_copy['detailed_household_summary_in_household'].map(mp)
        return df_copy

    def categorize_industry(self, df):
        '''
        Categorizes industry information.
        '''
        df_copy = df.copy()
        mp = {
            ' Not in universe or children': 0,
            ' Agriculture': 1,
            ' Forestry and fisheries': 1,
            ' Mining' : 2,
            ' Construction': 3,
            ' Manufacturing-durable goods' : 4,
            ' Manufacturing-nondurable goods': 4,
            ' Wholesale trade' : 5,
            ' Retail trade' : 5,
            ' Transportation': 6,
            ' Utilities and sanitary services': 6,
            ' Finance insurance and real estate': 8,
            ' Business and repair services' : 9,
            ' Other professional services' : 9,
            ' Education': 10,
            ' Medical except hospital': 10,
            ' Hospital services': 10,
            ' Public administration' : 11,
            ' Armed Forces': 12
            }
        df_copy['major_industry_code'] = df_copy['major_industry_code'].map(mp)
        return df_copy

    def categorize_person(self, df):
        '''
        Categorizes person information.
        '''
        df_copy = df.copy()
        mp_cit = {
        ' Native- Born in the United States': 1,
        ' Foreign born- Not a citizen of U S ': 5,
        ' Foreign born- U S citizen by naturalization': 4,
        ' Native- Born abroad of American Parent(s)': 3,
        ' Native- Born in Puerto Rico or U S Outlying': 2
        }
        mp_origin = {
        ' Native- Born in the United States': 1,
        ' Foreign born- Not a citizen of U S ': 2,
        ' Foreign born- U S citizen by naturalization': 2,
        ' Native- Born abroad of American Parent(s)': 1,
        ' Native- Born in Puerto Rico or U S Outlying': 1
        }

        df_copy['origin'] = df_copy['citizenship'].map(mp_origin)
        df_copy['citizenship'] = df_copy['citizenship'].map(mp_cit)

        return df_copy
    
    def drop_duplicates(self,df):
        '''
        Drops duplicates from the dataset.
        '''
        df_copy = df.copy()
        df_copy = df_copy.drop_duplicates(inplace=False)
        return df_copy

    def normalize(self, column, train, test):
        '''
        Normalizes all columns in the dataframe.
        '''
        train_copy = train.copy()
        test_copy = test.copy()

        train_values= train_copy[column].values.reshape(-1, 1)
        test_values = test_copy[column].values.reshape(-1, 1)
        
        # From sklearn
        min_max_scaler = preprocessing.MinMaxScaler()
        
        train_values_normalized = min_max_scaler.fit_transform(train_values)
        test_values_normalized = min_max_scaler.transform(test_values)
        
        
        train_copy[column] = train_values_normalized
        test_copy[column] = test_values_normalized

        return train_copy,test_copy 

    # Bring it all together in a processing pipeline
    def preprocess(self, df, df_test):
        '''
        Preprocess the dataframe using preprocess methods.
        '''

        continuous_columns = ['age','wage_per_hour','capital_gains','capital_losses',
                      'dividends_from_stocks','num_persons_worked_for_employer',
                      'instance_weight','weeks_worked_in_years',
                      'education','detailed_household_and_family_stat','class_of_work',
                      'marital_status','citizenship','detailed_household_summary_in_household']

        categorical_columns = list(set(df.columns) - set(continuous_columns))

        X_train, X_test = self.categorize_features(categorical_columns, df, df_test)
        # Education
        X_train = self.categorize_education(X_train)
        X_test = self.categorize_education(X_test)

        #Household information
        X_train =  self.categorize_household_info(X_train)
        X_test = self.categorize_household_info(X_test)
        
        # Work class
        X_train = self.categorize_work_class(X_train)
        X_test = self.categorize_work_class(X_test)
        
        # Marital status
        X_train = self.categorize_marital(X_train)
        X_test = self.categorize_marital(X_test)
        
        # Financial information
        X_train = self.categorize_financial_info(X_train)
        X_test = self.categorize_financial_info(X_test)

        X_train = self.categorize_household_summary(X_train)
        X_test  = self.categorize_household_summary(X_test)
        
        # Person
        X_train = self.categorize_person(X_train)
        X_test = self.categorize_person(X_test)
        

        X_train, X_test = self.normalize('instance_weight', X_train, X_test)
        # Drop duplicates
        X_train = self.drop_duplicates(X_train)
        X_test = self.drop_duplicates(X_test)

        return X_train, X_test




    
    
