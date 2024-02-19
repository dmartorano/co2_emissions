# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 5: Group Project  

## Table of Contents

### `README.md`
### `Exploratory_Data_Analysis_Folder`
- Project_5_New_EDA_ben.ipynb
- Project_5_EDA_ben.ipynb
- Project_5_EDA_Dom.ipynb
### `Modeling_Folder`
Ben files
Project_5_Modeling_Dom.ipynb
Daniel files
### `Visualizations_Folder`
Visualizations ben file
Visualizations Presentation files
### `Powerpoint_Presentation`

## General 

This project focuses on analyzing environmental data with social impact.

### Environmental Data from the Energy Information Administration and Kaggle scraped from the EPA and World Bank:

[Insert Sources Here]

### Problem statement:

Communities across the United States are facing the dangers of climate change and air pollution. To address the issue, we are surveying air pollution and renewable energy datasets to create predictive models that can be utilized by policymakers, economists, and civil society organizations. The goal of this research is to better aid our understanding on how air pollution is affecting our society as a whole. The model will be capable of predicting air pollution levels as a function of changes in renewable energy  and fossil fuel productions.    

---

## File Descriptions 

### `README.md`
### `Exploratory_Data_Analysis_Folder`
- Project_5_New_EDA_ben.ipynb: Includes data imports, cleaning, munging, and exploratory data analysis (EDA) of \text{CO_{2}} data and energy source data. Energy sources: coal, natural gas, nuclear energy, petroleum and renewable energy. Renewable energy includes solar, wind, hydroelectric, and geothermal. Units for \text{CO_{2}} data are (Per capita emissions in metric tons per person). Units for energy sources are British Thermal Units. Source: https://www.eia.gov/state/seds/seds-change/index.php/
- Project_5_EDA_ben.ipynb:
- Project_5_EDA_Dom - Took cleaned state-level data and generated cleaned national average data. Cleaned national average data for all variables used for various time series models throughout group.
### `Modeling_Folder`
- Regression Modeling file: Exploration of various regression models evaluated based on R^{2} score agreement amongst train and test sets and Root Mean Squared Error (RMSE). Best Regression models: Third: Linear Regression with Polynomial features: The train score was 0.9956 while test score was 0.9941 with RMSE values train 1.1213 and test 1.371. Second: K Nearest Neighbors with a train score of 1.0 and test score 0.99476 with RMSE values 0.0 for train and test 1.2920. First: Bagging Tree Regressor with train score 0.9989 and test score 0.9953 with RMSE values train 0.5586 and test 1.2230. 
Project_5_Modeling_Dom - Utilized clean national average data to generate EDA graphs for all variables and create a multivariate time series model. Multivariate model found that, after a small increase to CO2 emissions from their COVID era low, CO2 emissions are forecasted to continue in a downward trend based on energy consumption levels.

### `Visualizations_Folder`
### `Powerpoint_Presentation`

### Group Members:

Ben Moss, Daniel Kim, Dominic Martorano.

---

### Presentation Contents:

- Problem summary.
- Walkthrough of blueprint to find solution.
- Demonstration of solution.
- Summaries of models fit and their performance (Root Mean Squared Error). 
- Brief discussion of limitations in the process. (i.e. data collection issues, missing values)
- Discussion of next steps.
