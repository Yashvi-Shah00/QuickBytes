
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
import pandas as pd

from sklearn import preprocessing 
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics, neighbors
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from newspaper import Article
# %matplotlib inline

import os
import sys
#os.chdir('D:\School\Sem 7\Major project\Classification\Preprocessing')
sys.path.insert(1, 'D:\School\Sem 7\Major project\Classification\Preprocessing')
from normalization import normalize_corpus

"""Reading the data"""
df = pd.read_csv('D:\School\Sem 7\Major project\Classification\dataset\dataset.csv')

df.info()

"""Normalizing the data"""

df['FILTERED_CONTENT'] = df['CONTENT'].apply(normalize_corpus)

df.head()

# Encode labels in column 'category'.
label_encoder = preprocessing.LabelEncoder() 
df['CATEGORY']= label_encoder.fit_transform(df['CATEGORY']) 

sns.countplot(df.CATEGORY)
plt.xlabel('Category')
plt.ylabel('count')
plt.title('Before sampling')

#Sampling the data

df0 = df[df.CATEGORY==0]
df1 = df[df.CATEGORY==1]
df2 = df[df.CATEGORY==2]
df3 = df[df.CATEGORY==3]
# print('df.CATEGORY==0',df.CATEGORY==0)
# print('df.CATEGORY==1',df.CATEGORY==1)
# print('df0',df0)
# print('df1',df1)
# print('df2',df2)
# print('df3',df3)
# print('df.CATEGORY',df.CATEGORY)
samples = df.CATEGORY.value_counts().tolist()

df0 = resample(df0, 
                   replace=True,   
                   n_samples=samples[0], 
                   random_state=1130)
df1 = resample(df1, 
                   replace=True,    
                   n_samples=samples[0],
                   random_state=123)
df2 = resample(df2, 
                   replace=True,    
                   n_samples=samples[0],
                   random_state=123)
df3 = resample(df3, 
                   replace=True,    
                   n_samples=samples[0],
                   random_state=123)


df_sampled = pd.concat([df0,df1,df2,df3])

df= df_sampled



sns.countplot(df_sampled.CATEGORY)
plt.xlabel('Category')
plt.ylabel('count')
plt.title('After sampling')

training_set, test_set, training_labels, test_labels = train_test_split(df["FILTERED_CONTENT"], df["CATEGORY"], test_size=0.33, random_state=42)

"""Feature Extractiong using TF-IDF vector"""

#Tfidf 
tfidvectorizer = TfidfVectorizer(min_df=2, 
                                 ngram_range=(2,2),
                                 smooth_idf=True,
                                 use_idf=True)
tfid_train_features = tfidvectorizer.fit_transform(training_set)

tfid_test_features = tfidvectorizer.transform(test_set)

print("tfid_test_features",tfid_test_features)

"""Classification using TF-IDF features"""

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(tfid_train_features,training_labels)

predictions_SVM = SVM.predict(tfid_test_features)


print("Support Vector Machine using TF-IDF\n")
print("Accuracy: ",accuracy_score(predictions_SVM, test_labels)*100,"\n")

print("Classification Report\n")
print(classification_report(test_labels,predictions_SVM))


#Plotting the confusion matrix
cm = confusion_matrix(test_labels, predictions_SVM)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['business', 'entertainment','health','technology'],rotation =45); 
ax.yaxis.set_ticklabels(['business', 'entertainment','health','technology'],rotation =45);