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
            - lower case/accentss/etc
        - missing data?
        - tokenize, stem, lemmatized 
        - stopwords
    - Explore
        - Wordclouds
        - 
    - Model
    - Deliver


What does your MVP look like? What are your team's need to haves vs nice to haves?
Put answers to these questions in your standup doc (link in Google Classroom)
