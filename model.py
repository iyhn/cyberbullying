import pandas as pd
import string
import re
import sys
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from imblearn.over_sampling import SMOTE
from sklearn.externals.joblib import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

remove = '[' + string.punctuation + string.ascii_letters + string.digits + ']'
thai = re.compile(r'[\u0e00-\u0e7f]', re.UNICODE)
df_train = pd.read_csv('aaaa.csv')
print(df_train.shape)
df_train = df_train.sample(n=df_train.shape[0],random_state=5)

# df_train = df[df.label.str.contains('1|2|3|4')]
# tmp = df[df.label.str.contains('2,4')]
# df_train.label = df_train.label.replace('2,4','2')

# df_train = pd.concat([df_train,tmp])
# df_train.label = df_train.label.replace('2,4','4')

# df_train = pd.concat([df_train,df[df.label.str.contains('0')].sample(n=df_train.shape[0],random_state=1)])

y = df_train.label
y_tfidf = df_train.label
y = y.astype(int)

# vectorizer = CountVectorizer(tokenizer=lambda a: list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',a)), engine='newmm'))))
# tfidf_vect = TfidfVectorizer(tokenizer=lambda a: list(filter(str.strip,word_tokenize(u'{}'.format(re.sub(remove,'',a)), engine='newmm'))))
vectorizer = CountVectorizer(ngram_range=(2,2),tokenizer=lambda a: list(filter(str.strip,word_tokenize(u'{}'.format(''.join(thai.findall(a))), engine='newmm'))))
tfidf_vect = TfidfVectorizer(ngram_range=(2,2),tokenizer=lambda a: list(filter(str.strip,word_tokenize(u'{}'.format(''.join(thai.findall(a))), engine='newmm'))))

print("vertorizing")
X = vectorizer.fit_transform(df_train['text'].values.astype('U')).toarray()
print("tfidf-vertorizing")
X_tfidf = tfidf_vect.fit_transform(df_train['text'].values.astype('U')).toarray()
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=30)
# X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(X_tfidf,y,test_size=0.2, random_state=30)
# X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print("normal - Smoting")
X,y = SMOTE().fit_resample(X,y)
print("tfidf - Smoting")
X_tfidf, y_tfidf = SMOTE().fit_resample(X_tfidf, y_tfidf)




Encoder = LabelEncoder()
# y_train = Encoder.fit_transform(y_train)
# y_test = Encoder.fit_transform(y_test)

Naive = naive_bayes.MultinomialNB()
# Naive.fit(X_train,y_train)
Naive_tfidf = naive_bayes.MultinomialNB()
# Naive_tfidf.fit(X_tfidf_train,y_tfidf_train)

#Prediction_NB = Naive.predict(X_test)
#Prediction_NB_tfidf = Naive_tfidf.predict(X_tfidf_test)
print("Naive Bayes")
CrossValNaive = cross_val_score(Naive, X, y, cv=5)
print("CrossValNaive ->", CrossValNaive)
print(sum(CrossValNaive)/5)
CrossValNaiveTFIDF = cross_val_score(Naive_tfidf, X_tfidf, y_tfidf, cv=5)
print("CrossValNaiveTFIDF ->", CrossValNaiveTFIDF)
print(sum(CrossValNaiveTFIDF)/5)
# Naive.fit(X,y)
# Naive_tfidf.fit(X_tfidf,y)
#print("Naive Accuracy score -> ",accuracy_score(Prediction_NB,y_test)*100)
#print("Naive TF-IDF Accuracy score -> ",accuracy_score(Prediction_NB_tfidf,y_tfidf_test)*100)

SVM = svm.SVC(C=1.0, kernel='linear', gamma='auto')
# SVM.fit(X_train,y_train)
SVM_tfidf = svm.SVC(C=1.0, kernel='linear', gamma='auto')
# SVM_tfidf.fit(X_tfidf_train,y_tfidf_train)

#Prediction_SVM= SVM.predict(X_test)
#Prediction_SVM_tfidf= SVM_tfidf.predict(X_tfidf_test)
print("Support vector machine")
CrossValSVM = cross_val_score(SVM, X, y, cv=5)
print("CrossValSVM ->", CrossValSVM)
print(sum(CrossValSVM)/5)
CrossValSVMTFIDF = cross_val_score(SVM_tfidf, X_tfidf, y_tfidf, cv=5)
print("CrossValSVMTFIDF ->", CrossValSVMTFIDF)
print(sum(CrossValSVMTFIDF)/5)
SVM.fit(X,y)
SVM_tfidf.fit(X_tfidf,y)
#print("SVM Accuracy score -> ",accuracy_score(Prediction_SVM,y_test)*100)
#print("SVM TF-IDF Accuracy score -> ",accuracy_score(Prediction_SVM_tfidf,y_tfidf_test)*100)

rm = RandomForestClassifier(n_estimators=100,random_state=0)
# rm.fit(X_train,y_train)
rm_tfidf = RandomForestClassifier(n_estimators=100,random_state=0)
# rm_tfidf.fit(X_tfidf_train,y_tfidf_train)

#Prediction_RandomForest = rm.predict(X_test)
#Prediction_RandomForest_tfidf = rm_tfidf.predict(X_tfidf_test)
print("Random forest")
CrossValRandom = cross_val_score(rm, X, y, cv=5)
print("CrossValRandom ->", CrossValRandom)
print(sum(CrossValRandom)/5)
CrossValRandomTFIDF = cross_val_score(rm_tfidf, X_tfidf, y_tfidf, cv=5)
print("CrossValRandomTFIDF ->", CrossValRandomTFIDF)
print(sum(CrossValRandomTFIDF)/5)
rm.fit(X,y)
rm_tfidf.fit(X_tfidf,y)
# print("RandomForest Accuracy score -> ",accuracy_score(Prediction_RandomForest,y_test)*100)
# print("RandomForest TF-IDF Accuracy score -> ",accuracy_score(Prediction_RandomForest_tfidf,y_tfidf_test)*100)

logistic = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
# logistic.fit(X_train,y_train)
logistic_tfidf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
# logistic_tfidf.fit(X_tfidf_train,y_tfidf_train)

print("Logistic")
CrossValLogistic = cross_val_score(logistic, X, y, cv=5)
print("CrossValLogistic ->", CrossValLogistic)
print(sum(CrossValLogistic)/5)
CrossValLogisticTFIDF = cross_val_score(logistic_tfidf, X_tfidf, y_tfidf, cv=5)
print("CrossValLogisticTFIDF ->", CrossValLogisticTFIDF)
print(sum(CrossValLogisticTFIDF)/5)
logistic.fit(X,y)
logistic_tfidf.fit(X_tfidf,y)

#dump(Naive, 'Naive.joblib')
#dump(Naive_tfidf, 'Naive_tfidf.joblib')
#dump(SVM, 'SVM.joblib')
#dump(SVM_tfidf, 'SVM_tfidf.joblib')
#dump(rm, 'rm.joblib')
#dump(rm_tfidf, 'rm_tfidf.joblib')
#dump(vectorizer, 'vectorizer.joblib')
#dump(tfidf_vect, 'tfidf_vect.joblib')


# lr = LinearRegression().fit(X_train,y_train)
# lr_tfidf = LinearRegression().fit(X_tfidf_train,y_tfidf_train)
# Prediction_LinearRegression = lr.predict(X_test)
# Prediction_LinearRegression_tfidf = lr_tfidf.predict(X_tfidf_test)
# print("LinearRegression Accuracy score -> ",accuracy_score(Prediction_LinearRegression,y_test)*100)
# print("LinearRegression TF-IDF Accuracy score -> ",accuracy_score(Prediction_LinearRegression_tfidf,y_tfidf_test)*100)

model = {
    '0': Naive,
    '1': Naive_tfidf,
    '2': SVM,
    '3': SVM_tfidf,
    '4': rm,
    '5': rm_tfidf
}

feature = {
    '0': vectorizer,
    '1': tfidf_vect,
    '2': vectorizer,
    '3': tfidf_vect,
    '4': vectorizer,
    '5': tfidf_vect
}

type_model = ''
input_data = ''
while type_model != "exit":

    if(type_model != ''):
        selected_feature = feature[type_model]
        selected_model = model[type_model]
        print('กรอกข้อความที่ต้องการตรวจสอบ')
        input_data = sys.stdin.readline().strip()

        while input_data != "back" and input_data != 'exit':
            input_data = selected_feature.transform([input_data]).toarray()
            result = selected_model.predict(input_data)[0]
            if(result == 0) :
                print('ไม่ใช่ข้อความข่มเหงรังแก')
            elif(result == 1):
                print('การคุกคามทางเพศ')
            elif(result == 2):
                print('การโจมตีขู่ทำร้ายหรือถ้อยคำหยาบคาย')
            elif(result == 3):
                print('เชื้อชาติศาสนาและวัฒนธรรม')
            elif(result == 4):
                print('สติปัญญา​ฐานะและรูปลักษณ์ภายนอก')
            input_data = sys.stdin.readline().strip()

    if(input_data == 'exit'): break
    print('เลือกประเภท model ที่ต้องการใช้')
    print('0: Naive')
    print('1: Naive TFIDF')
    print('2: SVM')
    print('3: SVM TFIDF')
    print('4: RandomForest')
    print('5: RandomForest TFIDF')
    type_model = sys.stdin.readline().strip()
    if(type_model == 'exit'): break
    if(type_model == 'dump'):
        dump(Naive, 'Naive.joblib')
        dump(Naive_tfidf, 'Naive_tfidf.joblib')
        dump(SVM, 'SVM.joblib')
        dump(SVM_tfidf, 'SVM_tfidf.joblib')
        dump(rm, 'rm.joblib')
        dump(rm_tfidf, 'rm_tfidf.joblib')
        dump(vectorizer, 'vectorizer.joblib')
        dump(tfidf_vect, 'tfidf_vect.joblib')
    if(type_model not in model): type_model = ''
