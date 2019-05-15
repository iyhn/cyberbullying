# -*- coding: utf-8 -*-
import pandas as pd
import sys
import os
import string
import re
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

remove = '[' + string.punctuation + string.ascii_letters + string.digits + ']'
# a = [0,0,0,0,0]
# s = []
# df1 = pd.DataFrame()
# df2 = pd.DataFrame()
# df3 = pd.DataFrame()
# df4 = pd.DataFrame()
df_final = pd.DataFrame()

for file in os.listdir('/Users/ruby/Downloads/Archive 3/csv'):
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        df.label = df.label.fillna(0)
        if(df.label.dtypes == float): df.label = df.label.astype(int)
        df.label = df.label.astype(str)
        df_final = pd.concat([df_final,df],sort=True)
        
        # df1 = pd.concat([df1,df[df['label'].str.contains('1')]],sort=True)
        # df2 = pd.concat([df2,df[df['label'].str.contains('2')]],sort=True)
        # df3 = pd.concat([df3,df[df['label'].str.contains('3')]],sort=True)
        # df4 = pd.concat([df4,df[df['label'].str.contains('4')]],sort=True)
        
# df1['text_tokenized'] = [list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',i)), engine='newmm'))) for i in df1['text']]
# df2['text_tokenized'] = [list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',i)), engine='newmm'))) for i in df2['text']]
# df3['text_tokenized'] = [list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',i)), engine='newmm'))) for i in df3['text']]
# df4['text_tokenized'] = [list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',i)), engine='newmm'))) for i in df4['text']]

# for i in df_final.label:
#         print(type(i))
# df_final.label = df_final.label.astype(str)

df1 = df_final[df_final.label.str.contains('1')]
df2 = df_final[df_final.label.str.contains('2')]
df3 = df_final[df_final.label.str.contains('3')]
df4 = df_final[df_final.label.str.contains('4')]

xzc = pd.concat([df1,df2])
xzc = pd.concat([xzc,df3])
xzc = pd.concat([xzc,df4])
# xzc = pd.concat([xzc,df_final[df_final.label.str.contains('0')].sample(n=xzc.shape[0],random_state=1)])
y=xzc.label
y = y.replace('2,4', '4')
y = y.astype(int)
print(y)

# df1 = pd.concat([df1,df_final[~df_final.label.str.contains('1')].sample(n=df1.shape[0],random_state=1)])
# df1 = pd.concat([df2,df_final[~df_final.label.str.contains('2')].sample(n=df2.shape[0],random_state=1)])
# df1 = pd.concat([df3,df_final[~df_final.label.str.contains('3')].sample(n=df3.shape[0],random_state=1)])
# df1 = pd.concat([df4,df_final[~df_final.label.str.contains('4')].sample(n=df4.shape[0],random_state=1)])

# y1 = df1.label.replace(['2','3','4','2,4'],'0')
# y2 = df2.label.replace(['1','3','4'],'0')
# y3 = df3.label.replace(['2','1','4','2,4'],'0')
# y4 = df4.label.replace(['2','3','1'],'0')

# df_train = y[y.str.contains('1')]
# df_train = pd.concat([df_train,y[y.str.contains('0')].sample(n=399,random_state=1)])
# print(df_train)
# print(y[y.str.contains('1')])
# y = y.astype(int)

vectorizer = CountVectorizer(tokenizer=lambda a: list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',a)), engine='newmm'))))
tfidf_vect = TfidfVectorizer(tokenizer=lambda a: list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',a)), engine='newmm'))))
# X1 = vectorizer.fit_transform(df1['text'].values.astype('U')).toarray()
# X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2, random_state=60)
X2 = vectorizer.fit_transform(xzc['text'].values.astype('U')).toarray()
X2_tfidf = tfidf_vect.fit_transform(xzc['text'].values.astype('U')).toarray()
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y,test_size=0.2, random_state=60)
X2_tfidf_train, X2_tfidf_test, y2_tfidf_train, y2_tfidf_test = train_test_split(X2_tfidf,y,test_size=0.2, random_state=60)
# X3 = vectorizer.fit_transform(df3['text'].values.astype('U')).toarray()
# X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,test_size=0.2, random_state=60)
# X4 = vectorizer.fit_transform(df4['text'].values.astype('U')).toarray()
# X4_train, X4_test, y4_train, y4_test = train_test_split(X4,y4,test_size=0.2, random_state=60)

Encoder = LabelEncoder()
y2_train = Encoder.fit_transform(y2_train)
y2_test = Encoder.fit_transform(y2_test)

# Naive1 = naive_bayes.MultinomialNB()
# Naive1.fit(X1_train,y1_train)
Naive2 = naive_bayes.MultinomialNB()
Naive2.fit(X2_train,y2_train)
Naive2_tfidf = naive_bayes.MultinomialNB()
Naive2_tfidf.fit(X2_tfidf_train,y2_tfidf_train)
# Naive3 = naive_bayes.MultinomialNB()
# Naive3.fit(X3_train,y3_train)
# Naive4 = naive_bayes.MultinomialNB()
# Naive4.fit(X4_train,y4_train)

# svm1 = svm.SVC(C=1.0, kernel='linear', degree='3', gamma='auto')
# svm1.fit(X2_train,y2_train)

# Prediction_NB1 = Naive1.predict(X1_test)
Prediction_NB2 = Naive2.predict(X2_test)
Prediction_NB2_tfidf = Naive2_tfidf.predict(X2_tfidf_test)
# Prediction_NB3 = Naive3.predict(X3_test)
# Prediction_NB4 = Naive4.predict(X4_test)

# Prediction_SVM2 = Naive2.predict(X2_test)

# print("Naive 1 Acc score -> ",accuracy_score(Prediction_NB1,y1_test)*100)
print("Naive 2 Acc score -> ",accuracy_score(Prediction_NB2,y2_test)*100)
print("Naive TF-IDF 2 Acc score -> ",accuracy_score(Prediction_NB2_tfidf,y2_tfidf_test)*100)
# print("Naive 3 Acc score -> ",accuracy_score(Prediction_NB3,y3_test)*100)
# print("Naive 4 Acc score -> ",accuracy_score(Prediction_NB4,y4_test)*100)

# print("SVM 2 Acc score -> ",accuracy_score(Prediction_SVM2,y2_test)*100)

input_data = sys.stdin.readline().strip()
while input_data != "exit":
        input_data = vectorizer.transform([input_data]).toarray()
        print(Naive2.predict(input_data))
        # print(Naive2.predict(input_data))
        # print(Naive3.predict(input_data))
        # print(Naive4.predict(input_data))
        input_data = sys.stdin.readline().strip()

# classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
# classifier.fit(X_train,y_train)

# print(vectorizer.get_feature_names())

# For Export
# df1.to_csv('df1.csv',index=False)
# df2.to_csv('df2.csv',index=False)
# df3.to_csv('df3.csv',index=False)
# df4.to_csv('df4.csv',index=False)

# print("{} {} {} {}".format(df1.shape[0],df2.shape[0],df3.shape[0],df4.shape[0]))

# df = pd.read_csv('โง่.csv')
# df.label = df.label.astype(str)
# df1 = pd.concat([df1,df[df['label'].str.contains('2')]])
# print(df1)

# 99 70 1 69