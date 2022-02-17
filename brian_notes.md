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
    - Prep
        - clean
            - lower case/accents/etc
        - missing data?
        - tokenize, stem, lemmatized 
        - stopwords
    - Explore
        - Wordclouds
        - Clustering? Is there any cluster that makes intuitive sense?
    - Model
        - model sentiment analysis over time
        - Model speaker prediction
        - Iterate models with varying hyperparameters
    - Deliver


What does your MVP look like? What are your team's need to haves vs nice to haves?


Put answers to these questions in your standup doc (link in Google Classroom)


## Initial Questions?
    - Is it possible to find differing levels of sentiment at different times? For different hosts? Programs?
    - Do differing levels of sentiment relate to real-world news happenings? e.g. 9/11; Iraq/Afghanistan wars; presidential campaigns (with an eye to 2016 in particular); 
    - Can we identify a news category for given utterances? -> This applies particularly depending on the type of aggregation we are able to accomplish
    - Are there words that are said more frequently by hosts? By time of day? By category?
    - What host(s) say(s) the most words? 
    - 