from matplotlib.pyplot import title
from psaw import PushshiftAPI
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer # tokenize words
from nltk.corpus import stopwords
nltk.download('vader_lexicon') # get lexicons data
nltk.download('punkt') # for tokenizer
nltk.download('stopwords')
import numpy as np
import datetime
import psycopg2
import psycopg2.extras
api = PushshiftAPI()


start_time = int(datetime.datetime(2022, 1, 30).timestamp())

submissions = api.search_submissions(after=start_time,
                                     subreddit='SatoshiStreetBets',
                                     filter=['title', 'submission','post'])
posts = []
#ml_subreddit = reddit.subreddit('MachineLearning')
for post in submissions:
    if "BTC" in post.title: 
        print(post.title)
        posts.append([post.title, post.created])
posts = pd.DataFrame(posts,columns=['title', 'created'])
#print(posts)
 
sid = SentimentIntensityAnalyzer()                            
posts.to_csv("btc posts.csv")
res = [*posts['title'].apply(sid.polarity_scores)]
print(res[:3])
sentiment_df = pd.DataFrame.from_records(res)
news = pd.concat([posts, sentiment_df], axis=1, join='inner')
news.head()

THRESHOLD = 0.2

conditions = [
    (news['compound'] <= -THRESHOLD),
    (news['compound'] > -THRESHOLD) & (news['compound'] < THRESHOLD),
    (news['compound'] >= THRESHOLD),
    ]

values = ["neg", "neu", "pos"]
news['label'] = np.select(conditions, values)
news.to_csv("btc postsnn.csv") 
#for submission in submissions:
 #   words = submission.title.split()
  #  print(words)
