# NPR Sentiment Analysis

## This file/repo contains information on performing NLP analyses on NPR Interviews to find sentimentality.

## Project Description

This Jupyter Notebook and presentation explore the NPR interview corpus. The data used relates to NPR radio interviews from 1999 to 2019. An important aspect of NPR as an independent media outlet is to effectively convey information to create a more informed public; we intend to build a machine learning model that interpets a corpus of interviews to predict sentiment scores.

Accuracy (for classification) will be used as the primary metric for evaluation; many models will be built using different features and hyperparameters to find the model of best fit.  One of the final deliverables will be the RMSE value resulting from the best model, comparable with the baseline accuracy.

Additionally, a Jupyter Notebook with our findings and conclusions will be a key deliverable; many .py files will exist as support for the main Notebook (think "under-the-hood" coding that will facilitate the presentation).


## The Plan

The intention of this project is to follow the data science pipeline by acquiring and wrangling the relevant information from the NPR Media Dialog Transcripts dataset, located at: https://www.kaggle.com/shuyangli94/interview-npr-media-dialog-transcripts. Manipulation of the data to a form suitable to the exploration of variables, application in machine learning, and to graphically present the most outstanding findings, including actionable conclusions, along the way.

## Project Goals

The ultimate goal of this project is to build a model that predicts whether an utterance was said by a host or by a guest.  Additional goals include studying sentiment over time (including attempting to predict sentiment) and an exploration of the utterances, hosts and stories that were broadcast on NPR over the years in question.

## Data Dictionary

| variable      | meaning       |
| ------------- |:-------------:|
| sentiment | our target variable, created by sentiment analysis of NPR interviews |
| episode | each observation in our data is representative of each interview |
| episode_id | identification number for each episode |
| episode_date | date the episode was uploaded |
| episode_order | order the interview occurred |
| program | name of NPR program |
| headline | article's headline |
| is_host | whether the speaker is speaking as a host |
| utterance | utterance text |
| speaker | individual that is speaking during the utterance | 
| title | name of the interview/article | 


## Initial Questions?
### to MVP
    - Is it possible to find differing levels of sentiment at different times? For different hosts? Programs?
    - Are there words that are said more frequently by hosts? By time of day? By category?
    - What host(s) say(s) the most words?
    - Is there a difference in the mean sentiment by speaker? Program? etc
    - Applied statistics-> i.e. stats testing. Is there a difference in the mean sentiment by speaker? Program? etc 
    - Are there stopwords, outside of the standard, that could be removed to help identify the speaker?
    
### and Beyond!
    - Do differing levels of sentiment relate to real-world news happenings? e.g. 9/11; Iraq/Afghanistan wars; presidential campaigns (with an eye to 2016 in particular)
    - Can we identify a news category for given utterances? -> This applies particularly depending on the type of aggregation we are able to accomplish

##  Steps to Reproduce

The files required to reproduce are availabe at:
https://www.kaggle.com/shuyangli94/interview-npr-media-dialog-transcripts
The only files used in this study were "utterrances.csv", and "episodes.csv".

As an alternative, considering the huge size of the files involved, a copy of the final .csv resulting from all wrangling can be downloaded at https://drive.google.com/file/d/1RANtH1IS5iDmlUOigKMMG8EQOgh6dkNp/view?usp=sharing.

All other necessary files can be cloned along with the rest of the project at the project's github:
https://github.com/npr-nlp/npr-nlp-project

Once the repository is cloned and the final csv downloaded, the notebook should run with no issues.

<!-- LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
-- viz.py: used for creating several graphics for my final presentation
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- feature_engineering.py: contains functions to help choose the 'best' features using certain sklearn functions  -->

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
- Ordinal encode the bathrooms and bedrooms -->

## A Note on the Source Data

- The dataset used in this project is available at https://www.kaggle.com/shuyangli94/interview-npr-media-dialog-transcripts?select=headlines.csv, and was compiled by Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley. A detailed description of its background and collection can be read at https://arxiv.org/pdf/2004.03090.pdf.


