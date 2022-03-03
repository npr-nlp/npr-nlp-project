# all the imports used throughout the notebook:
import json
import pandas as pd
import numpy as np
import wrangle
import acquire
from prepare import basic_clean, tokenize, lemmatize, stem, remove_stopwords, prep_string_data#, split_data
import scipy as sp
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Create an empty evaluation dataframe
eval_df = pd.DataFrame(columns=['Model_Type', 'Train_Accuracy','Validate_Accuracy','Accuracy_Difference', 'Beats_Baseline_By'])


# function to store accuracy for comparison purposes
def append_eval_df(model_type, train_accuracy, validate_accuracy):
    d = {'Model_Type': [model_type],'Train_Accuracy':[train_accuracy], \
        'Validate_Accuracy': [validate_accuracy], \
            'Accuracy_Difference': [train_accuracy - validate_accuracy], \
                'Beats_Baseline_By':[validate_accuracy - (1-0.37535714285714283)]}  # let's try to do this programatically
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def run_modeling():
    # Create an empty evaluation dataframe
    eval_df = pd.DataFrame(columns=['Model_Type', 'Train_Accuracy','Validate_Accuracy','Accuracy_Difference', 'Beats_Baseline_By'])


    # Define the df
    df = wrangle.get_npr_data()
    # Add two columns that weren't where expected
    df['question_mark_count'] = df.utterance.str.count(r"[\?]")  # counts question marks
    df['utterance_word_count'] = df.utterance.apply(str.split).apply(len) # length of  utterance

    # sample down due to size issuees
    small_df = df.sample(100_000, random_state=222)
    
    ############################# THIS BIT TURNED OUT NOT TO BE REQUIRED #########################
    # counts of observations per speaker
    counts = small_df['speaker'].value_counts()
    # limiting our df to only speakers with 3 or more utterances. This helped wheen stratifying the splits
    rest = small_df[~small_df['speaker'].isin(counts[counts < 3].index)]
    ############################# BUT IT'S GOOD CODE FOR REFERENCE ###############################

    # get initial splits based no "rest", which is a sampled-down df with speakers with 2 or less observations removed
    train, validate, test = wrangle.split_data(small_df)
    print(f"train shape is {train.shape}")
    print(f"validate shape is {validate.shape}")
    print(f"test shape is {test.shape}")



    # define the features to go into the model
    features = train.drop(columns = ['date','story_id_num','speaker','utterance','program','title',\
        'is_host','clean','lemmatized']).columns.to_list()

    # X_ and y_ splits
    X_train = train[features]
    y_train = train.is_host

    X_validate = validate[features]
    y_validate = validate.is_host

    X_test = test[features]
    y_test = test.is_host
    print("Basic Splits Created")

    # add a column to each basic split for the "actual" result (i.e. y_)
    train['actual'] = y_train
    validate['actual'] = y_validate
    test['actual'] = y_test

    # create splits including TF-IDF features
    # assign the tfidf vectorizer
    tfidf = TfidfVectorizer()
    # fit/transform vectorizer on train only
    X_train_tfidf = tfidf.fit_transform(train.lemmatized) 
    X_validate_tfidf = tfidf.transform(validate.lemmatized)
    X_test_tfidf = tfidf.transform(test.lemmatized)
    # add features to X_tfidf splits
    X_train_tfidf_plusfeatures = sp.sparse.hstack((X_train_tfidf, pd.DataFrame(X_train.question_mark_count),\
        pd.DataFrame(X_train.utterance_order),pd.DataFrame(X_train.vader),pd.DataFrame(X_train.utterance_word_count)))
    X_validate_tfidf_plusfeatures = sp.sparse.hstack((X_validate_tfidf, pd.DataFrame(X_validate.question_mark_count),\
        pd.DataFrame(X_validate.utterance_order),pd.DataFrame(X_validate.vader),pd.DataFrame(X_validate.utterance_word_count)))
    X_test_tfidf_plusfeatures = sp.sparse.hstack((X_test_tfidf, pd.DataFrame(X_test.question_mark_count),\
        pd.DataFrame(X_test.utterance_order),pd.DataFrame(X_test.vader),pd.DataFrame(X_test.utterance_word_count)))
    print('TF_IDF Splits Created')
    # Crate splits with count vectorized features
    # define the model #cell 156  in  the  noteebook
    cv = CountVectorizer()
    # fit/transform vectorizer on train only
    X_train_cv = cv.fit_transform(train.lemmatized) 
    X_validate_cv = cv.transform(validate.lemmatized)
    X_test_cv = cv.transform(test.lemmatized)
    # add features to X_cv splits
    X_train_cv_plusfeatures = sp.sparse.hstack((X_train_cv, pd.DataFrame(X_train.question_mark_count),\
        pd.DataFrame(X_train.utterance_order),pd.DataFrame(X_train.vader),pd.DataFrame(X_train.utterance_word_count)))
    X_validate_cv_plusfeatures = sp.sparse.hstack((X_validate_cv, pd.DataFrame(X_validate.question_mark_count),\
        pd.DataFrame(X_validate.utterance_order),pd.DataFrame(X_validate.vader),pd.DataFrame(X_validate.utterance_word_count)))
    X_test_cv_plusfeatures = sp.sparse.hstack((X_test_cv, pd.DataFrame(X_test.question_mark_count),\
        pd.DataFrame(X_test.utterance_order),pd.DataFrame(X_test.vader),pd.DataFrame(X_test.utterance_word_count)))
    print('Count Vectorizer Splits Created')

    # create baseline on mode (is_host == False)
    train['baseline_pred'] = False
    validate['baseline_pred'] = False
    test['baseline_pred'] = False

    # append to eval df
    ev1 = append_eval_df('baseline_pred', accuracy_score(train.is_host, train.baseline_pred),accuracy_score(validate.is_host, validate.baseline_pred))

    #######################
    # LOGISTIC REGRESSION #
    #######################
    print("Logistic Regression Beginning")
    # Only features
    ###############
    # definee model; fit on X_ and y_train
    lm = LogisticRegression().fit(X_train, y_train)
    # add corresponding predictions
    train['predicted_X_just_features'] = lm.predict(X_train)
    validate['predicted_X_just_features'] = lm.predict(X_validate)
    test['predicted_X_just_features'] = lm.predict(X_test)
    # append t oeval df
    ev2 = append_eval_df('Log_Reg_Just_Features', accuracy_score(train.actual,train.predicted_X_just_features),\
            accuracy_score(validate.actual,validate.predicted_X_just_features))
    # TF-IDF alone
    ##############
    # define model
    lm_tfidf = LogisticRegression().fit(X_train_tfidf, y_train)
    # add pred column
    train['predicted_Xtfidf'] = lm_tfidf.predict(X_train_tfidf)
    validate['predicted_Xtfidf'] = lm_tfidf.predict(X_validate_tfidf)
    test['predicted_Xtfidf'] = lm_tfidf.predict(X_test_tfidf)
    # append to eval df
    ev3 = append_eval_df('Log_Reg_Just_TFIDF', accuracy_score(train.actual, train.predicted_Xtfidf),\
                            accuracy_score(validate.actual, validate.predicted_Xtfidf))
    #  TF-IDF Plus Features
    #######################
    # define model
    lm_tfidf_plusfeatures = LogisticRegression().fit(X_train_tfidf_plusfeatures, y_train)
    # prdiction column
    train['predicted_Xtfidf_plusfeatures'] = lm_tfidf_plusfeatures.predict(X_train_tfidf_plusfeatures)
    validate['predicted_Xtfidf_plusfeatures'] = lm_tfidf_plusfeatures.predict(X_validate_tfidf_plusfeatures)
    test['predicted_Xtfidf_plusfeatures'] = lm_tfidf_plusfeatures.predict(X_test_tfidf_plusfeatures)
    # append to eval df
    ev4 = append_eval_df('Log_Reg_TFIDF_Plus_Features', accuracy_score(train.actual, train.predicted_Xtfidf_plusfeatures),\
                            accuracy_score(validate.actual, validate.predicted_Xtfidf_plusfeatures))
    # Count Vectorizer Alone
    ################################
    # model
    lm_cv = LogisticRegression().fit(X_train_cv, y_train)
    # prediction columns
    train['predicted_X_cv'] = lm_cv.predict(X_train_cv)
    validate['predicted_X_cv'] = lm_cv.predict(X_validate_cv)
    test['predicted_X_cv'] = lm_cv.predict(X_test_cv)
    # append eval df
    ev5 = append_eval_df('Log_Reg_CV', accuracy_score(train.actual, train.predicted_X_cv),\
                            accuracy_score(validate.actual, validate.predicted_X_cv))
    # Count Vectorizer Plus Features
    ################################
    #model
    lm_cv_plusfeatures = LogisticRegression().fit(X_train_cv_plusfeatures, y_train)
    #prediction columns
    train['predicted_X_cv_plus_features'] = lm_cv_plusfeatures.predict(X_train_cv_plusfeatures)
    validate['predicted_X_cv_plus_features'] = lm_cv_plusfeatures.predict(X_validate_cv_plusfeatures)
    test['predicted_X_cv_plus_features'] = lm_cv_plusfeatures.predict(X_test_cv_plusfeatures)
    # append eval df
    ev6 = append_eval_df('Log_Reg_CV_Plus_Features', accuracy_score(train.actual, train.predicted_X_cv_plus_features),\
                            accuracy_score(validate.actual, validate.predicted_X_cv_plus_features))
    print("LR done")
    #######################
    #### DECISION TREE ####
    #######################
    print('Decision Tree beginning')
    clf = DecisionTreeClassifier(max_depth = 5, random_state = 222)
    # Only features
    ###############
    # define model
    clf_just_features = clf.fit(X_train, y_train)
    # pred cols
    train['predicted_clf_just_features'] = clf.predict(X_train)
    validate['predicted_clf_just_features'] = clf.predict(X_validate)
    # append eval df
    ev7 = append_eval_df('Decision_Tree_Just_Features', accuracy_score(train.actual, train.predicted_clf_just_features),\
                            accuracy_score(validate.actual, validate.predicted_clf_just_features))
    # TF-IDF alone
    ##############
    # define model
    clf_tfidf = clf.fit(X_train_tfidf, y_train)
    # pred cols
    train['predicted_clf_tfidf'] = clf_tfidf.predict(X_train_tfidf)
    validate['predicted_clf_tfidf'] = clf_tfidf.predict(X_validate_tfidf)
    test['predicted_clf_tfidf'] = clf_tfidf.predict(X_test_tfidf)
    # append eval df
    ev8 = append_eval_df('DecisionTree_Just_TFIDF', accuracy_score(train.actual, train.predicted_clf_tfidf),\
                            accuracy_score(validate.actual, validate.predicted_clf_tfidf))
    #  TF-IDF Plus Features
    #######################
    # define model
    clf_tfidf_plus_features = clf.fit(X_train_tfidf_plusfeatures, y_train)
    # pred cols
    train['CLF_predicted_Xtfidf_plusfeatures'] = clf_tfidf_plus_features.predict(X_train_tfidf_plusfeatures)
    validate['CLF_predicted_Xtfidf_plusfeatures'] = clf_tfidf_plus_features.predict(X_validate_tfidf_plusfeatures)
    test['CLF_predicted_Xtfidf_plusfeatures'] = clf_tfidf_plus_features.predict(X_test_tfidf_plusfeatures)
    #append eval df
    ev9 = append_eval_df('Decision_Tree_TFIDF_PlusFeatures', accuracy_score(train.actual, train.CLF_predicted_Xtfidf_plusfeatures),\
                            accuracy_score(validate.actual, validate.CLF_predicted_Xtfidf_plusfeatures))
    # Count Vectorizer Alone
    ################################
    # model
    clf_countvectorizer = clf.fit(X_train_cv, y_train)
    # pred cols
    train['predicted_Xcv'] = clf_countvectorizer.predict(X_train_cv)
    validate['predicted_Xcv'] = clf_countvectorizer.predict(X_validate_cv)
    test['predicted_Xcv'] = clf_countvectorizer.predict(X_test_cv)
    # append eval df
    ev10 = append_eval_df('DecisionTree_Just_Countvectorizer', accuracy_score(train.actual, train.predicted_Xcv),\
                            accuracy_score(validate.actual, validate.predicted_Xcv))
    # Count Vectorizer Plus Features
    ################################
    #model
    clf_countvectorizer_plus_features = clf.fit(X_train_cv_plusfeatures, y_train)
    # pred cols
    train['predicted_Xcv_plus_features'] = clf_countvectorizer_plus_features.predict(X_train_cv_plusfeatures)
    validate['predicted_Xcv_plus_features'] = clf_countvectorizer_plus_features.predict(X_validate_cv_plusfeatures)
    test['predicted_Xcv_plus_features'] = clf_countvectorizer_plus_features.predict(X_test_cv_plusfeatures)
    # append eval df
    ev11 = append_eval_df('DecisionTree_CountVectorizere_PlusFeatures', accuracy_score(train.actual, train.predicted_Xcv_plus_features),\
                            accuracy_score(validate.actual, validate.predicted_Xcv_plus_features))
    print('DT done')

    #######################
    #### RANDOM FOREST ####
    #######################
    print('Random Forest beginning')
    rf = RandomForestClassifier(min_samples_leaf=3,criterion='gini',max_depth=2,random_state=222)
    # Only features
    ###############
    # define model
    rf_just_features = rf.fit(X_train, y_train)
    # pred  cols
    train['predicted_rf_just_features'] = rf_just_features.predict(X_train)
    validate['predicted_rf_just_features'] = rf_just_features.predict(X_validate)
    #  append eval df
    ev12 = append_eval_df('RandomForest_Just_Features', accuracy_score(train.actual, train.predicted_rf_just_features),\
                            accuracy_score(validate.actual, validate.predicted_rf_just_features))
    # TF-IDF alone
    ##############
    # define model
    rf_tfidf = rf.fit(X_train_tfidf, y_train)
    # pred cols
    train['predicted_rf_tfidf'] = rf_tfidf.predict(X_train_tfidf)
    validate['predicted_rf_tfidf'] = rf_tfidf.predict(X_validate_tfidf)
    test['predicted_rf_tfidf'] = rf_tfidf.predict(X_test_tfidf)
    # append eval df
    ev13 = append_eval_df('RandomForest_Just_TFIDF', accuracy_score(train.actual, train.predicted_rf_tfidf),\
                            accuracy_score(validate.actual, validate.predicted_rf_tfidf))
    #  TF-IDF Plus Features
    #######################
    # define model
    rf_tfidf_plus_features = rf.fit(X_train_tfidf_plusfeatures, y_train)
    # pred cols
    train['RF_predicted_Xtfidf_plusfeatures'] = rf_tfidf_plus_features.predict(X_train_tfidf_plusfeatures)
    validate['RF_predicted_Xtfidf_plusfeatures'] = rf_tfidf_plus_features.predict(X_validate_tfidf_plusfeatures)
    test['RF_predicted_Xtfidf_plusfeatures'] = rf_tfidf_plus_features.predict(X_test_tfidf_plusfeatures)
    # append eval df
    ev14 = append_eval_df('RandomForest_TFIDF_PlusFeatures', accuracy_score(train.actual, train.RF_predicted_Xtfidf_plusfeatures),\
                            accuracy_score(validate.actual, validate.RF_predicted_Xtfidf_plusfeatures))
    # Count Vectorizer Alone
    ################################
    # model
    rf_countvectorizer = rf.fit(X_train_cv, y_train)
    # pred cols
    train['predicted_Xcv'] = rf_countvectorizer.predict(X_train_cv)
    validate['predicted_Xcv'] = rf_countvectorizer.predict(X_validate_cv)
    test['predicted_Xcv'] = rf_countvectorizer.predict(X_test_cv)
    # append eval df
    ev15 = append_eval_df('RandomForest_Just_CountVectorizer', accuracy_score(train.actual, train.predicted_Xcv),\
                            accuracy_score(validate.actual, validate.predicted_Xcv))

    # Count Vectorizer Plus Features
    ################################
    #model
    rf_countvectorizer_plus_features = rf.fit(X_train_cv_plusfeatures, y_train)
    #pred cols
    train['predicted_Xcv_plus_features'] = rf_countvectorizer_plus_features.predict(X_train_cv_plusfeatures)
    validate['predicted_Xcv_plus_features'] = rf_countvectorizer_plus_features.predict(X_validate_cv_plusfeatures)
    test['predicted_Xcv_plus_features'] = rf_countvectorizer_plus_features.predict(X_test_cv_plusfeatures)
    # append eval df
    ev16 = append_eval_df('RandomForest_CountVectorizer_Plus_Features', accuracy_score(train.actual, train.predicted_Xcv_plus_features),\
                            accuracy_score(validate.actual, validate.predicted_Xcv_plus_features))
    print('RF done')

    ######################################
    ###### DISPLAY EVAL DF ###############
    ######################################
    eval_df = eval_df.append(ev1).append(ev2).append(ev3).append(ev4).append(ev5).append(ev6).append(ev7)\
    .append(ev8).append(ev9).append(ev10).append(ev11).append(ev12).append(ev13).append(ev14).append(ev15).append(ev16)
    return eval_df, train, validate, test