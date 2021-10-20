import csv
import random
import re
import codecs

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import matplotlib.pyplot as plt
import numpy as np

from collections import Counter
import nltk;nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import TextBlob

SIA= SentimentIntensityAnalyzer()

positive=0
negative=0
neutral=0
total=0

hashtags=[]
print("Performing sentiment analysis...")
for i in range(5):
    print(".",end="")
    time.sleep(1)

filepath="UNGA.csv"
with codecs.open(filepath,"r",encoding='utf-8',errors='ignore') as csvfile:
    reader=csv.reader(csvfile)
    tweetsList=[]
    cleanTweetsList=[]
    for row in reader:
        tweet=row[1].strip()
        cleanTweet=" ".join(re.findall("[a-zA-Z]+",tweet))
        analysisVS = SIA.polarity_scores(cleanTweet)
        #analysisTB = TextBlob(cleanTweet)
        tweetsList.append(tweet)
        cleanTweetsList.append(cleanTweet)
        
        total=total+1
        if(analysisVS['compound'] < -0.05):
         negative = negative+1
        elif(analysisVS['compound'] > 0.05):
         positive = positive+1
        else:
         neutral = neutral+1
         
        
print("total", total)
print("negative", negative)
print("positive",positive)
print("neutral",neutral)        
        
randomTweets=[]
randomCleanTweets=[]
randomNumber=random.sample(range(1,200),5)
index=0

for i in range(5):
    number=random.randint(1,200)
    randomTweets.append(tweetsList[randomNumber[index]])
    randomCleanTweets.append(tweetsList[randomNumber[index]])
    index = index+1
for tweet in randomCleanTweets:
    print()
    print(tweet,end='')
    analysisVs = SIA.polarity_scores(tweet)
    print("=>",analysisVS)
    
finalcount={}
for i in tweetsList:
    hashtags.append(re.findall(r"#(\w+)",i))
hashtagnew=[item for sub in hashtags for item in sub]
counts =Counter(hashtagnew)
counts = dict(counts)
finalcount= dict(sorted(counts.items(),key = lambda kv:kv[1],reverse = True))
countname = list(finalcount.keys())

objects = ('Positive','Neutral','Negative')
y_pos = np.arange(len(objects))
performance = [positive,neutral,negative] 

plt.bar(y_pos, performance, align = 'center', alpha=0.5) 
plt.xticks(y_pos,objects)
plt.ylabel('# of tweets')
plt.title('Twitter Sentiment Analysis-(Bar Graph) \n')
plt.show()

colors = ['yellowgreen','gold','orangered']
explode = (0, 0, 0.1)
plt.pie(performance, explode = explode, labels = objects, colors=colors,
        autopct='%1.1f%%', shadow= False, startangle = 140)
plt.axis('equal')
plt.title('Twitter Sentiment Analysis-(Pie Chart) \n')
plt.show()

x = np.arange(len(finalcount))
y = list(finalcount.values())
x= x[:15]
y= y[:15]
countname = countname[:15]
plt.bar(x, y)
plt.title('Most Trending Hashtags\n')
plt.xticks(x, countname, rotation='vertical')
plt.ylabel('Number of tweets')
plt.xlabel('#Hashtags')
plt.tight_layout()
plt.show()                             
    
    