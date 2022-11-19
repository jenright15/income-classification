# Income Classification Project

[Income Classification Project](#income-classification-project)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Background Information](#background-information)
  - [Data](#data)
    - [Features](#features)
  - [Methodology](#methodology)
    - [EDA](#eda)
    - [Data Preperation](#data-preperation)
    - [Data Modeling](#data-modeling)
    - [Model Assessment & Selection](#model-assessment--selection)
    - [Results & Conclusions](#results--conclusions)
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
- Statistics
-  Number of instances training data = 199523
   -  Duplicate or conflicting instances : 46716
-  Number of instances in test data = 99762
   -  Duplicate or conflicting instances : 20936
-  Distribution
   -  Probability for the label '- 50000' : 93.80%
   -  Probability for the label '50000+' : 6.20%
- Number of features = 40 (continuous : 7 nominal : 33)

### Data Preperation

### Data Modeling

### Model Assessment & Selection

### Results & Conclusions

## Usage

## Requirements

