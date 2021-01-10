# MBTIPredict

## Objective

Try to predict MBTI type based on posts data from https://www.kaggle.com/datasnaek/mbti-type. Since there is only one given feature and label, all of the features used in the model are generated through NLP analysis of the posts data.

## Data Analysis

All analysis was done in Google Colab, with my personal Google Drive mounted to the file system.

Based on the theory of the MBTI, there are 16 personality types. Therefore, this is a multivariate multiclass classification task.

However, the given data does not have an equal distribution of entries for each personality type, so stratified train-test data splitting must be used for best representation of each class:
| Type | Row Count |
| ---- | --------- |
| INFP | 1832 |
| INFJ | 1470 |
| INTP | 1304 |
| INTJ | 1091 |
| ENTP | 685 |
| ENFP | 675 |
| ISTP | 337 |
| ISFP | 271 |
| ENTJ | 231 |
| ISTJ | 205 |
| ENFJ | 190 |
| ISFJ | 166 |
| ESTP | 89 |
| ESFP | 48 |
| ESFJ | 42 |
| ESTJ | 39 |

Features that have been generated during analysis using TextBlob library (https://textblob.readthedocs.io/en/dev/index.html#):
- Average post sentence length in characters
- Standard deviation of post sentence length in characters
- Average post sentence number of words
- Standard deviation of post sentence number of words
- Average post sentence descriptor length in characters
- Standard deviation of post sentence descriptor length in characters
- Words that each personality type uses relatively more frequently than other personality types (see `Determining the Distinguishing Words for each Personality Type` in `.ipynb` file)
- Sentiment polarity of posts
- Sentiment subjectivity of posts
- Count of words in posts which are considered to have "extreme" polarity (|polarity| >= some threshold)
- Count of words in posts which are considered to have "extreme" subjectivity (|subjectivity| >= some threshold)
- Count of words in posts belong to certain classes of words ("mood", "profanity", "irony")
- Count of punctuation marks (e.g. ',', '.', '!', '?') used in posts

All features have been included together for analysis and feature importance determined by a random forest classifier has allowed me to arbitrarily exclude features (see `.ipynb` file for features marked with tag # UF).

Since generating some of the features takes a significant amount of time, I generated the features and saved them in pickle files, found under the GitHub folder `preprocessed_data`. In the `.ipynb` code, the paths are relative to my Google Drive file system, but are easily adaptable to new filesystems. Simply loading the pickle files and swapping in and out features to include saves a lot of time from repeated feature computation.

## Model

Scikit-Learn's implementations of `SVC`, `RandomForestClassifier`, and `LogisticRegressor` have all achieved around 62% accuracy. Using the ensemble `VotingClassifier`, which aggregrates the votes based on soft voting (i.e. averaging class probabilities), the accuracy has increased to around 65%. However, due to the imbalance of data representating each personality type and the possibility that personality types cannot be distinguished solely on website posts, I believe 65% is the best that I can achieve.
