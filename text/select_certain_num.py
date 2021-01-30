# -*- coding: utf-8 -*-
import random
from syntok.tokenizer import Tokenizer

fr = open('./train_data', 'r', encoding='utf-8')
sentences = fr.read().split('\n\n')
random.shuffle(sentences)

num = 100
selected = []
tokenizer = Tokenizer()
for sent in sentences:
    entity, gloss = sent.split('\n')
    if len(entity) > 25:
        continue
    if len(list(tokenizer.tokenize(gloss))) > 45:
        continue
    selected.append(entity + '\n' + gloss)
    if len(selected) == num:
        break
    
fw = open('./train_%d'%num, 'w', encoding='utf-8')
fw.write('\n\n'.join(selected))
fw.close()
