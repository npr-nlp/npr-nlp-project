## Elevator Pitch: we aim to do a study 20 years of NPR news transcripts to get an idea of the overall sentiment of NPR news over the years.  Along the way, we intend to pair major news events to the sentiment of news stories, and do a "sentiment forecast" model for unseen Data; we also expect some other interesting exploration, including predictions on whether the speaker is an interviewer or interviewee, and potentially, what news category a story belongs to.  Future iterations may include other exploration and modeling, including adding outside news sources and predicting fake vs. legitimate news.


## What is your goal?: Find out how the mood of the nation has evolved over twenty years based on public radio news broadcasts. Build a predictive model for mood at a given point in time; also predict who the speaker is; what program it belongs to; and possibly what news category.

## What is your target? 
    - Predict host/not_host. Predict sentiment (tsa)

## What is one observation?
    - We need to explore to see if aggregation is possible on a story/article basis, or other bases

## What is going to take place in each stage of the pipeline? What is the minimally viable amount of work you can do in each stage?
    - Acquisition: 
        - merge utterances with the episodes csv. and maybe the headlines too.
        - (find another dataset of fake news? pot-mvp)
        - include the steps and even the dates the files were downloaded in the 'Reproducibility' section
    - Prep
        - clean
            - lower case/accents/etc
        - missing data?
        - tokenize, stem, lemmatized 
        - stopwords
        - bigrams
        - include a MD file explaining clean/prep? 
        - do we need to do more prep to clearly label hosts vs. non-hosts?
        - POS (parts of speech analysis)? This is going to be unrealistic to have time for
        - Is utterance a worthwhile feature? scale it
    - Explore
        - Wordclouds-> break down by year? or for the time periods around big events?
        - Clustering? Is there any cluster that makes intuitive sense?  Number of speakers in a given article, for instance?
        - Anomalies? Unexpected sentiment analysis results? 
        - Applied statistics-> i.e. stats testing. Is there a difference in the mean sentiment by speaker? Program? etc
        - Are there stopwords, outside of the standard, that could be removed to help identify the speaker?
        - Possibility of leaving the "laughtrack" observations in and identifying those utterances that are followed by a laughtrack->from here, model to discover what is meant to be a joke
        - vader on different corpi (lemmatized, stemmed, stopped, etc.)
        - What was the most negative newstory in the bunch?  The most positive? This will make for great explo and is easy to get to: aggregate on 'episode_id' (which is the 'article')
        - The most negative and positive individual program? Aggregating on program and date (and individual program is a progran name on a given date)
    - Model
        - model sentiment analysis over time
        - Model speaker prediction
        - Iterate models with varying hyperparameters
        - Explore the possibility of applying topic modeling--this would necessarily be unsupervised machine learning, and may be outside the scope of our project (UNLESS...we find a labeled data set we could run against the model for validation and testing...again, way past mvp for this step)--> see NLP packages Gensim and spaCy
        - Don't forget that the countvectorizer package has a function to pull bigrams!! (ngram_range = (x, y))

    - Deliver
        - Definitely some nerdy audio to open things up with.
        - Executive summary; all the headings in the Readme (be clear what is meant by 'recommendations' versus 'next steps')
        - Polished notebook and helper files
        - A little organization in the final repo--at a minimum, all files that aren't the final notebook and helper files could be stashed in a separate folder


What does your MVP look like? What are your team's need to haves vs nice to haves?


Put answers to these questions in your standup doc (link in Google Classroom)


## Initial Questions?
    - Is it possible to find differing levels of sentiment at different times? For different hosts? Programs?
    - Do differing levels of sentiment relate to real-world news happenings? e.g. 9/11; Iraq/Afghanistan wars; presidential campaigns (with an eye to 2016 in particular); 
    - Can we identify a news category for given utterances? -> This applies particularly depending on the type of aggregation we are able to accomplish
    - Are there words that are said more frequently by hosts? By time of day? By category?
    - What host(s) say(s) the most words? 
    - Is there a difference in the mean sentiment by speaker? Program? etc




## Moving forward things:
    - remove oddball speakers (where text has found its way into the speaker box)
    - Do a better job d
