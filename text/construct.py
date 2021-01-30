# -*- coding: utf-8 -*-
import re
fr = open('./train_split_selected', 'r', encoding='utf-8')
sentences = fr.read().split('\n\n')

constructed = []
for sentence in sentences:
    sent, sent_m = sentence.split('\n')
    entity_len = len(re.search(r'的+', sent_m).group())
    entity = sent[:entity_len]
    gloss = sent[entity_len:]
    gloss = re.sub(' *\(.*\)', '', gloss)
    constructed.append( entity + '\n' + gloss.strip())
    
fw = open('./train_data', 'w', encoding='utf-8')
fw.write('\n\n'.join(constructed))
fw.close()