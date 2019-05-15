import string
import re
import sys
from sklearn.externals.joblib import load
from pythainlp import word_tokenize

remove = '[' + string.punctuation + string.ascii_letters + string.digits + ']'
thai = re.compile(r'[\u0e00-\u0e7f]', re.UNICODE)

vectorizer = load('vectorizer.joblib')
tfidf_vect = load('tfidf_vect.joblib')
Naive = load('Naive.joblib')
Naive_tfidf = load('Naive_tfidf.joblib')
SVM = load('SVM.joblib')
SVM_tfidf = load('SVM_tfidf.joblib')
rm = load('rm.joblib')
rm_tfidf = load('rm_tfidf.joblib')

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
    if(type_model not in model): type_model = ''