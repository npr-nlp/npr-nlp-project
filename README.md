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
| story_id_num | identification number for each episode |
| utterance_order | turn number within episode |
| speaker | individual that is speaking during the utterance |
| utterance | utterance text |
| program | name of NPR program |
| title | name of the interview/article | 
| is_host | our target variable, whether the speaker is speaking as a host |
| clean | cleaned version of utterance column |
| lemmatized | lemmatized version of utterance column |
| vader | created by sentiment analysis of NPR interviews |
| date | date the episode was uploaded |
| question_mark_count | feature created out of the number of question marks in the corpus |
| utterance_word_count | feature created out of the number of words in the corpus |
| actual |  |
| baseline_pred |  |
| predicted_X_just_features |  |
| predicted_Xtfidf |  |
| predicted_Xtfidf_plusfeatures |  |
| predicted_X_cv |  |
| predicted_X_cv_plus_features |  |
| predicted_clf_just_features |  |
| predicted_clf_tfidf |  |
| CLF_predicted_Xtfidf_plusfeatures |  |
| predicted_Xcv |  |
| predicted_Xcv_plus_features |  |
| predicted_rf_just_features |  |
| predicted_rf_tfidf |  |
| RF_predicted_Xtfidf_plusfeatures |  |


## Missing values:
    - Dropped observations before 2005 due to large amounts of missing data
    - A very small number (less than 10) missing values after that were dropped
    - For modeling, speakers with two or less utterances were dropped to avoid issues splitting

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
The only files used in this study were "utterrances.csv", and "episodes.csv", which we downloaded on 2/2/22.

As an alternative, considering the huge size of the files involved, a copy of the final .csv resulting from all wrangling can be downloaded at https://drive.google.com/file/d/1RANtH1IS5iDmlUOigKMMG8EQOgh6dkNp/view?usp=sharing

A csv with the dataframe aggregated by story is at the following link, and will also be required:
https://drive.google.com/file/d/1IGRwXQw_ruGanvQycfQRuX7KfaeuV5Hk/view?usp=sharing

All other necessary files can be cloned along with the rest of the project at the project's github:
https://github.com/npr-nlp/npr-nlp-project

Installations required: NTLK; Gensim; spaCy; pyLDAvis; pickle package; and the standard Python libraries.

With these installations complete, and once the repository is cloned and the final csv downloaded, the notebook should run with no issues.

<!-- LIST OF MODULES USED IN THE PROJECT, FOUND IN THE PROJECT DIRECTORY:
-- wrangle.py: for acquiring, cleaning, encoding, splitting and scaling the data.  
-- viz.py: used for creating several graphics for my final presentation
-- model.py: many, many different versions of the data were used in different feature selection and modeling algorithms; this module is helpful for splitting them up neatly.
-- feature_engineering.py: contains functions to help choose the 'best' features using certain sklearn functions  -->

## Key findings, recommendations and takeaways
    
We were able to get great answers from the data in our exploration--demonstrating, for example, the keen adherence to neutrality shown by NPR hosts in their utterances. Other exploration did not evolve into clear takeaways, except to say it was not predictive. This is in reference to the analysis of sentiment over time, which, although some anecdotal trends seem to exist, on the whole is generally very full of noise and difficult to draw conclusions from. 

The models we built were interesting and yielded worthwhile results--when predicting whether an utterance was said by a guest or host, our best model beat the baseline by over 23%. This seems a good start for, at a minimum, filtering out probable fake utterances.  

## Recommendations

We recommend our model be used as an analytical tool of news articles. NPR, for instance, may find this exploration useful for analytical reasons (do we need to address imbalances in tone between programs? hosts? over time?), but another stakeholder is the public writ large, which has an interest in understanding the content being broadcast by the largest not-for-profit news source on the radio waves. The classification power of our final model is also a quote actually belonds to an NPR correspondent.

## Next steps

This datset provides ample opportunities for exploration, and any number could be suggested, such as:
- Network diagrams for the relationship between hosts and programs
- Radial tree for hosts and top words said
- Topic modeling
- Evaluating our Prophet model for the time series analysis (maybe we have a prediction of future sentiment after all?)
- Small issues that occured when lemmatizing
- Continued trimming by adding stopwords
- Expore the effect of the question mark count--is this an example of target leakage? Or just low-hanging fruit?
- Coninuing exploration of the corpuses such as Part of Speech analysis
- Use different word vector models for the LDA analysis-> GoogleNews-vectors, different spaCy models
- Etc.

## A Note on the Source Data

- The dataset used in this project is available at https://www.kaggle.com/shuyangli94/interview-npr-media-dialog-transcripts?select=headlines.csv, and was compiled by Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley. A detailed description of its background and collection can be read at https://arxiv.org/pdf/2004.03090.pdf.


