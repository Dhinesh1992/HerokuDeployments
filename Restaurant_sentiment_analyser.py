#!/usr/bin/env python
# coding: utf-8

# In[149]:


# Importing required models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

import pickle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[150]:


df = pd.read_csv('Restaurant_Reviews.tsv',sep='\t')
df.head()


# In[151]:


df.shape


# In[152]:


# Tokenizing the 'Reviews'
tokens = [word_tokenize(sentence) for sentence in df['Review']]
tokens


# In[153]:


# Removing the stopwords and non alphanumerics
new_tokens=[]
for lst in tokens:
    new_tokens.append([word for word in lst if word.casefold() not in stopwords.words('english') and word.isalnum()])
    
new_tokens


# In[154]:


# Stemming the the words 
#Initializing SnowballStemmer
ps =  PorterStemmer()

stemmed_review = []
for sentence in new_tokens:
    stemmed_review.append(' '.join([ps.stem(word) for word in sentence]))
    
stemmed_review


# In[155]:


# Creating bag of words using CountVectorizer
cv = CountVectorizer()
transformed_tokens = cv.fit_transform(stemmed_review).toarray()

transformed_tokens


# In[156]:


transformed_tokens.shape


# In[157]:


# Instantiating naive bayes model and splitting train and test data
nb_model = MultinomialNB(alpha=0.1)

cross_val_score(nb_model, transformed_tokens, df['Liked'], scoring='accuracy', cv=5).mean()


# #### Using SnowballStemmer transformed values as it gives more accuracy

# In[167]:


X_train, X_test, y_train, y_test = train_test_split(transformed_tokens, df['Liked'], test_size=0.2)


# In[168]:


y_train.value_counts(normalize=True)


# In[169]:


y_test.value_counts(normalize=True)


# In[170]:


nb_model.fit(X_train,y_train)

y_pred = nb_model.predict(X_test)


# In[171]:


from sklearn.metrics import classification_report, accuracy_score, f1_score


# In[172]:


accuracy_score(y_test,y_pred)


# In[173]:


f1_score(y_test,y_pred)


# In[174]:


print(classification_report(y_test,y_pred))


# In[175]:


# Exporting the countvectorizer and Naivebayes model

pickle.dump(cv, open('CountVectorizer.pkl','wb'))
pickle.dump(nb_model, open('NBmodel.pkl','wb'))


# In[100]:


nb_model.predict(cv.transform(["Good reviews"]).toarray())


# In[127]:


transformer = pickle.load(open('CountVectorizer.pkl','rb'))
model = pickle.load(open('NBmodel.pkl','rb'))


# In[146]:


model.predict(transformer.transform(['service was not bad']).toarray())[0]


# In[145]:


inp = 'service was not bad as i thought'
w = word_tokenize(inp)
s = [i for i in w if i.casefold() not in set(stopwords.words('english')) and i.isalpha()]
st = ' '.join([ps.stem(i) for i in s])
model.predict(transformer.transform([st]).toarray())


# In[ ]:




