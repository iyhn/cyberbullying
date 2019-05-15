import pandas as pd
import re
import os
from pythainlp.spell import *

#print(spell("เตงแค่นิ้วอ้นใช่รึป่ะ เตง สารรูป กับเค้ามา"))

#df = pd.DataFrame()
#c = 0
#a = 0
#for file in os.listdir("/Users/cyh/Downloads/Archive 3 2/csv"):
#    if(file.endswith(".csv") and file != 'final_data.csv'):
#        readFile = pd.read_csv(file)
#        readFile.label.fillna(0,inplace=True)
#        readFile.label = readFile.label.replace('2,4', 2)
#        readFile.label = readFile.label.astype('int64')
#        readFile.label = readFile.label.astype('str')
#        tmp = readFile[~readFile.label.str.contains('0')]
#        # print(tmp.shape[0])
#        tmp = pd.concat([
#            tmp,
#            readFile[readFile.label.str.contains('0')].head(tmp.label.shape[0])],
#            ignore_index=True)
#         # print(tmp.shape[0])
#        df = pd.concat([df, tmp], ignore_index=True,sort=True)
#
#
#df.to_csv('aaaa.csv',index=False)
#print(df.shape)

#df = pd.read_csv('aaaa.csv')
#df['label'].fillna(0,inplace=True)
##print(df.label)
#print('การคุกคามทางเพศ = {}'.format(df[df.label.str.contains('1')].shape[0]))
#print('การโจมตีขู่ทำร้ายหรือถ้อยคำหยาบคาย = {}'.format(df[df.label.str.contains('2')].shape[0]))
#print('เชื้อชาติศาสนาและวัฒนธรรม = {}'.format(df[df.label.str.contains('3')].shape[0]))
#print('สติปัญญา​ฐานะและรูปลักษณ์ภายนอก = {}'.format(df[df.label.str.contains('4')].shape[0]))
df = pd.read_csv('aaaa.csv')
df.label = df.label.astype('str')

print(df[df.label.str.contains('1|2|3|4')].shape)
print(df[df.label == '0'].shape)
print(df[df.label == '1'].shape)
print(df[df.label == '2'].shape)
print(df[df.label == '3'].shape)
print(df[df.label == '4'].shape)
