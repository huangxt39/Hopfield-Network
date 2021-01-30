# -*- coding: utf-8 -*-
import re

fr = open('./train_split_truncated', 'r', encoding='utf-8' )
sentences = fr.read().split('\n\n')

selected_sents = []

for sentence in sentences:
    sent, sent_m = sentence.split('\n')
    if sent_m[0] == '的':
        if re.search(r'[^A-Za-z0-9.,\)\(;\- ]', sent) is None:
            selected_sents.append(sent + '\n' + sent_m)

fw = open('./train_split_selected', 'w', encoding='utf-8')
fw.write('\n\n'.join(selected_sents))
fw.close()