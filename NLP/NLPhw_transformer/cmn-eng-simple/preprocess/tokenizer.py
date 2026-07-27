import os
import re
from opencc import OpenCC
import nltk
from nltk.tokenize import word_tokenize
from zhon.hanzi import punctuation

cc = OpenCC('t2s')
en, cn = [], []
with open('cmn.txt', 'r', encoding='utf-8') as f:
    for line in f:
        sentence = re.split('\t', line)
        sentence = list(filter(None, sentence))
        en_sentence = ''
        for word in word_tokenize(sentence[0]):
            en_sentence += word.lower() + ' '
        en.append(en_sentence)
        cn_sentence = ''
        for char in sentence[1].strip():
            if char in (' ', '\n', '\t', '\r'):
                continue
            cn_sentence += cc.convert(char) + ' '
        cn.append(cn_sentence)

with open('en.txt', 'w', encoding='utf-8') as f:
    for sentence in en:
        print (sentence, file=f)

with open('cn.txt', 'w', encoding='utf-8') as f:
    for sentence in cn:
        print (sentence, file=f)

print ('Tokenizer Done!')
