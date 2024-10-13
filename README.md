# disaster_tweets
Repo for the Kaggle competition "Natural Language Processing with Disaster Tweets"

https://www.kaggle.com/competitions/nlp-getting-started/overview

## Installation and setup

```
conda create -n disaster_tweets python=3.12
conda activate disaster_tweets
pip install -e .[dev]
python -m spacy download en
```

## Download competition data

```
cd data
kaggle competitions download -c nlp-getting-started
```
