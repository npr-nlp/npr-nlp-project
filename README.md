# REGRESSION

## This file/repo contains information related to my Regression project, using thee Zillow dataset from the Codeup database.

## Project Description

This Jupyter Notebook and presentation explore the Zillow dataset from the Codeup database. The data used relates to 2017 real estate transactions relating to single family homes in three California counties, and different aspects of the properties. An important aspect of the Zillow business model is to be able to publish accurate home values; I intend to build a machine learning model that makes predictions about the tax value of the homes in question.

I will use Residual Mean Square Error as my metric for evaluation; many models will be built using different features and hyperparameters to find the model of best fit.  One of the final deliverables will be the RMSE value resulting from my best model, contrasted with the baseline RMSE.

Additionally, a Jupyter Notebook with my main findings and conclusions will be a key deliverable; many .py files will exist as a back-up to the main Notebook (think "under-the-hood" coding that will facilitate the presentation).


## The Plan

The intention of this project is to follow the data science pipeline by acquiring and wrangling the relevant information from the Codeup database using a MySQL query; manipulating the data to a form suitable to the exploration of variables and machine learning models to be applied; and to graphically present the most outstanding findings, including actionable conclusions, along the way.

## Project Goals

The ultimate goal of this project is to build a model that predicts the tax value of the homes in question with a higher accuracy than the baseline I have chosen--the purpose being to be able to produce better, more accurate home price predictions than Zillow's competitors. 

## Initial Questions

- Are larger homes valued higher?  

- What other aspects can be identified about higher-value homes?

- Do more bathrooms relate to higher tax value? What about square feet per bathroom--is there a sweet spot?

- Newer homes are larger; they are also valued more highly. Is there an exception to the rule?


##  Steps to Reproduce

In  the case of this project, there are several python files that can be used to acquire, clean, prepare and otherwise manipulate the data in advance of exploration, feature selection, and modeling (listed below).

I split the data into X_train and y_train data sets for much of the exploration and modelling, and was careful that no features were directly dependent on the target variable (tax_value).  I created a couple of features of my own, which produced some useful insights, and dropped rows with null values (my final dataset was 44,864 rows long, from 52,442 that were downloaded using SQL)

Once the data is correctly prepared, it can be run through the sklearn preprocessing feature for polynomial regressiong and fit on the scaled X_train dataset, using only those features indicated from the recursive polynomial engineering feature selector (also an sklearn function).  This provided me with the best results for the purposes of my project.

LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
-- viz.py: used for creating several graphics for my final presentation
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- feature_engineering.py: contains functions to help choose the 'best' features using certain sklearn functions 

## Data Dictionary

Variable	Meaning
___________________
- bedrooms:	The number of bedrooms
- bathrooms:	The number of bathrooms
- sq_ft:	How many square feet
- tax_value:	The tax value of the home
- county:	What county does it belong to
- age:	How old is it?
- sq_ft_per_bathroom:	How many square feet per bathroom?
- LA:	Belongs to Los Angeles county
- Orange:	Belongs to Orange county
- Ventura:	Belings to Ventura county

Variables created in the notebook (explanation where it helps for clarity):

- zillow_sql_query (The original query that is copied to 'zillow' DataFrame for manipulation and analysis)
- zillow (A copy of the above DataFrame for use in the notebook)
- train
- validate
- test
- X_train
- y_train
- X_validate
- y_validate
- X_test
- y_test
- train_scaled
- X_train_scaled
- y_train_scaled
- validate_scaled
- X_validate_scaled
- y_validate_scaled
- test_scaled
- X_test_scaled
- y_test_scaled
- X_train_kbest (scaled dataframe with only KBest features)
- X_validate_kbest (scaled dataframe with only KBest features)
- X_test_kbest (scaled dataframe with only KBest features)
- X_train_rfe (scaled dataframe with only RFE features)
- X_validate_rfe (scaled dataframe with only RFE features)
- X_test_rfe (scaled dataframe with only RFE features)

Missing values: there were only something around 200 missing values in the data; thus, I have dropped them in the wrangle.py file due to their relative scarcity.  By removing outliers, several thousand rows were dropped.

## Key findings, recommendations and takeaways
    
Many of my initial questions had clear answers, but through statistical analysis I was able to make safe assumptions, which in turn led me down paths that I may have otherwise missed. More square feet, bedrooms and bathrooms all correlate to higher tax value; who would have thought that the relationship between these factors works differently depending on your county? 

Even my model, which has room for improvement, was capable of predicting the tax value of homes--by over 10 percent on the test data, and with similar margins for the train and validate data. I expect it to perform as well on unseen data as well--keeping certain parameters constant.

## Recommendations

I recommend exploring the relationship between half bathrooms and tax value as an easy addition to the model that may bring some benefit.  Additionally, running the model on LA county as separate from Ventura and orange might see some benefit, seeing as there area some different ways that the features work on tax value there. Square feet per bathroom is an example here--it didn't make it into my model, but may have value for future models.  

## Next steps

Continuing to select new features from the Codeup database stands to improve the model--assuming they don't present unforeseen problems such as numerous null values, etc.  Fireplaces and pools and certain feature engineering around those elements in particular could be interesting (is it detrimental to have too many? for example). Even from my most preliminary analysis, it seemed that half baths were beneficial, and it would be interesting to look at similar phenomena in those other featurs mentioned (and unmentioned).  I would also like to explore the method I've used to eliminate outliers, at least for some features--I'm worried that there might have been some useful info that was dropped with the outliers, especially one bedroom homes.

The following is a brief list of items that I'd like to add to the model:

- Incorporate a "has half bath" feature
- Run models on other features, including the half-bath and sfpb
- Run models on LA versus other counties
- Pull in other features from SQL
- Ordinal encode the bathrooms and bedrooms





