#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import pandas as pd
import re


# In[105]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[106]:


import nltk
nltk.download('stopwords')


# In[107]:


print(stopwords.words('english')) #they don't add much weigtage to our datasets


# In[108]:


df=pd.read_csv('train.csv')


# In[109]:


df.shape


# In[110]:


df.head()


# In[111]:


df.isnull().sum() #we have enough data set to train our model, no need of preprocessing


# In[112]:


df=df.fillna('') #replacing missing values with null strings


# In[113]:


df['content']=df['title']+' '+df['author']


# In[114]:


x=df.drop(columns='label',axis=1)
y=df['label']
print(x)
print(y)


# In[115]:


#do Stemming ie reducing a word to its root word ie removing all prefix and suffixes
#eg for actor,actress,acting --> act(reducing to accurate root word)
port_stem=PorterStemmer()
def stemming(content):
    stem_cont=re.sub('[^a-zA-Z]',' ',content) #removes unwanted numbers and commas from data(they would be replaced by as space)
    stem_cont=stem_cont.lower()
    #converting all the alphabets to lower case
    stem_cont=stem_cont.split()
    print(stem_cont)
    print("\n")
    stem_cont=[port_stem.stem(word) for word in stem_cont if not word in stopwords.words('english')] #removing the stopwords
    print(stem_cont)
    stem_cont=' '.join(stem_cont)
    return stem_cont
    


# In[116]:


df['content']=df['content'].apply(stemming)


# In[117]:


print(df['content'])


# In[118]:


X=df['content'].values
Y=df['label'].values


# In[119]:


print(type(X)) #1->fake news and 0->real news


# In[123]:


vect=TfidfVectorizer() #it counts the number of times a word is repeating in a paragraph and then tells its importance by its frequency
#Tf stands for term frequency and idf stands for inverse doc frequency
#word repeated several times loses meaning and hence termed as fake news
X=vect.fit_transform(X)


# In[124]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=4) #stratify is used to spread fake and real news in equal proportion
logreg=LogisticRegression()


# In[125]:


logreg.fit(x_train,y_train)


# In[126]:


from sklearn.metrics import accuracy_score
pred=logreg.predict(x_test)
acc=accuracy_score(pred,y_test)
print(acc)


# In[128]:


x_new=x_test[5]
pred=logreg.predict(x_new)
print(pred)
if(pred==0):
    print("real")
else:
    print("fake")


# In[ ]:




