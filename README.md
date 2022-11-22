# Income Classification Project

[Income Classification Project](#income-classification-project)
- [Income Classification Project](#income-classification-project)
  - [Description](#description)
  - [Background Information](#background-information)
  - [Data](#data)
  - [Methodology](#methodology)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Data Preperation](#data-preperation)
    - [Data Modeling](#data-modeling)
    - [Model Assessment \& Selection](#model-assessment--selection)
    - [Results \& Conclusions](#results--conclusions)
  - [Usage](#usage)
  - [Requirements](#requirements)

## Description
This repository's goal is for identifying characteristics that are assocatiated with an individual making more or less than $50,000 a year. This repository contains all code, documentation, data attributed to the project. 

## Background Information
The United States Census Bureau leads the country’s Federal Statistical System; its primary
responsibility is to collect demographic and economic data about America to help inform
strategic initiatives. Every ten years, the census is conducted to collect and organize information
regarding the US population to effectively allocate billions of dollars of funding to various
endeavors (e.g., the building and maintenance of hospitals, schools, fire departments,
transportation infrastructure, etc.). Additionally, the collection of census information helps to
examine the demographic characteristics of subpopulations across the country.

## Data
Sample dataset from US Census archive for ~ 300,000 individuals. The archive contains 4 files in ``/data/``
1. census_income_learn.csv (data for model training).
2. census_income_test.csv (data for model testing).
3. census_income_metadata.txt (metadata for both datasets).
4. census_income_additional_info.pdf (supplemental information).

## Methodology
1. Exploratory Data Analysis
    - Numerical and/or graphical representation of the data
2. Data Preparation
    - Cleaning, preprocessing, feature engineering
3. Data Modeling
    - Construction of machine learning models
4. Model Assessment
    - Model selection process based on evaluation of model performance
5. Results & Conclusions
    - Key findings and recommendations

### Exploratory Data Analysis
- After initial research into the data, the distribution of > 50,000 : < 50,000 is favored towards one class heavily (93% is > 50,000). This could introduce problems in classification methods and the performance metrics to be evaluated. There are also several dupicates and conflicting instances, which need to be resolved.
- The dataset is also unlabeled. 
- The train test split for this dataset is 2/3, 1/2 in a stratified fashion. 
- The feature ``instance_weight`` is purposefuly ignored, this is the number of people in the population that each record represents. 
- The ``Not in Universe`` attribute for some of these features indicates that the survee was not in the population for the category of the question.
- There is a large number of '?' values in the records of the features
  - `migration_code_change_in_msa` : 99696
  - `migration_code_change_in_reg` : 99696
  - `migration_code_move_within_reg` : 99696
  - `migration_prev_res_in_sunbelt` : 99696
  - `country_of_birth_father`: 6713 
  - `country_of_birth_mother`: 6119
  - `country_of_birth_self` : 3393
  
Statistics
-  Number of instances training data = 199523
   -  Duplicate or conflicting instances : 46716
-  Number of instances in test data = 99762
   -  Duplicate or conflicting instances : 20936
-  Distribution
   -  Probability for the label '- 50000' : 93.80%
   -  Probability for the label '50000+' : 6.20%
- Number of features = 40 (continuous : 10 nominal : 33)

### Data Preperation
After identifying data types and general statistics, the training and test data is cleaned with  `Preprocessing.py`
In this module:
- The columns are labeled
- Duplicates are dropped
- Categorical columns are encoded by class/group
- Continuous columns with high standard deviations are scaled and normalized
  - `age`
  - `wage_per_hour`
- Financial features are transformed into new features based on their value thresholds
  - `dividends_from_stocks` --> `has_stock`
  - `capital_losses` --> `has_losses`
  - `capital_gains` --> `has_gains`
- `instance_weight` and old financial features are dropped from the dataset as they are redundant or contain irrelavant information

To run this module:
```
from Preprocessing import IncomePreprocess

ip  = IncomePreprocess()
X_train, X_test = ip.preprocess(train_data, test_data)
```


### Data Modeling
Two types of classifcation algorithms are fittied to the data:
- Logisitic Regression
- Random Forests

There are several other algorithms that could be used, but for the sake of time, only these 2 were implemented.

Each algorithm is fitted to the training data and evaluated on the test set. 


### Model Assessment & Selection

The evaluation metrics used for the classifiers include:
- accuracy score
- precision 
- recall  
- f1-score
- roc auc 

Each model generates a classification report representative of each metric. Also, the AUC curve is plotted after training.

Because the classses are imbalanced in the dataset, we must be careful about choosing which metric to use. The benchmark for each model is the roc auc score. This metric represents the likelihood of of the model distinguishing between the two classes.

The values of the fitted models can be found in `../reports`

The logistic regression and random forest model perform similarly. 

The important features are selected from the fitted model and transformed the trainin data. The models are reevaluated on this engineered data. This process reduced only slightly changed the performance metrics from before. This process could be looped with hyperparameter tuning of each model, which in turn should improve the performance.

### Results & Conclusions

## Usage
In the directory `./code/`
Run the following command
```
pip install -r requirements.txt
```
Then run each cell of the jupyter notebook `Data_Modeling.ipynb`

## Requirements
```
seaborn==0.12.1
scikit-learn==1.1.3
pandas==1.3.2
matplotlib==3.6.2
graphviz==0.16
numpy==1.22.4
```

