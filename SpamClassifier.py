################################################################################################
# SPAM CLASSIFICATION PROBLEM STATEMENT
# Problem statement data is picked from https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
# IT IS EXECUTED USING STEMMING TECHNIQUE
# Accuracy obtained is 98.62
#
################################################################################################

## Data Processing Libraries
import pandas as pd
## NLP libraries
import nltk
## nltk.download() all libraries are downloaded
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re  ## for regular exression

##Read data
messages = pd.read_csv('smsspamcollection/SMSSpamCollection' , sep='\t' , names=['label' , 'message'])

ps = PorterStemmer()  ## initialize the stemmer

corpus = []

## Data Cleansing
for i in range(0 , len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i]) ## remove everything except of letters a to z and replace with blank
    review = review.lower()
    review = review.split()

    review= [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


## creating Bag Of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)  ### max_features - to select only important columns/words which are more frequent
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

## Train & testing split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.30 , random_state=0)

### Tranining with Naive Bayes classifier -- Naive bayes works more efficiently for NLP problems
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train , y_train)

y_pred = spam_detect_model.predict(X_test)

## Check confusion matrix
from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(y_test , y_pred)

## Check Accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test , y_pred)

### Print Outcome
print(confusion_m)

print('------------------------------')
print(accuracy)


