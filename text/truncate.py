# -*- coding: utf-8 -*-

from segtok.segmenter import split_single

sentences = [doc for doc in open('./train_split_216', "r", encoding="utf-8").read().split('\n\n') if doc]

print(f"read text file with {len(sentences)} lines")

truncated_sents = []

for sentence in sentences:
    string, string_m = sentence.split('\n')

    # print(string[:1000])
    sent_pair = []
    sents = [sent for sent in split_single(string) if sent][:2]
    sents_m = []
    _start = 0
    for sent in sents:
        sent_m = string_m[_start:_start+len(sent)]
        sents_m.append(sent_m)
        try:
            assert len(sent)==len(sent_m)
            assert all(sent[i]==sent_m[i] or sent_m[i]=='的' for i in range(len(sent)))
        except AssertionError:
            print(sent)
            print(sent_m)
            raise RuntimeError()
        _start += (len(sent)+1)
        #print(_start)
        assert len(string)==_start-1 or string[_start-1]==' '

    sent_pair = list(zip(sents, sents_m))
    
    if len(sent_pair)==0:
        continue
    string, string_m = tuple(zip(*sent_pair))
    truncated_sents.append( ' '.join(string) + '\n' + ' '.join(string_m) )
    
fw = open('./train_split_truncated', 'w', encoding='utf-8')
fw.write('\n\n'.join(truncated_sents))
fw.close()