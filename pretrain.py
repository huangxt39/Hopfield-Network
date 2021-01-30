# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from syntok.tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

class dictionary():
    def __init__(self):
        self.id2word = []
        self.word2id = {}
        self.add_word('<bos>')
        self.add_word('<eos>')
        self.add_word('<pad>')
        
    def add_word(self, word):
        if word not in self.id2word:
            self.word2id[word] = len(self.id2word)
            self.id2word.append(word)
        return self.word2id[word]
            
    def get_id(self, word):
        return self.word2id[word]
    
    def get_word(self, id):
        return self.id2word[id]
    
    def get_len(self):
        return len(self.id2word)

class wiki_dataset(Dataset):
    def __init__(self, data_path, num_of_data=100):
        super(wiki_dataset, self).__init__()
        
        char_dic = dictionary()
        word_dic = dictionary()
        tok = Tokenizer()
        max_entity_len = 25
        max_gloss_len = 45
        
        fr = open(data_path, 'r', encoding='utf-8')
        sents = fr.read().split('\n\n')
        source_ids = []
        target_ids = []
        for sent in sents:
            source, target = sent.split('\n')
            source = list(map(lambda x: char_dic.add_word(x), list(source))) #ids
            assert len(source) <= max_entity_len
            source.extend([char_dic.get_id('<pad>')]*(max_entity_len - len(source)))
            source_ids.append(source)
            
            target = list(map(lambda x: x.value, tok.tokenize(target))) #text
            target = list(map(lambda x: word_dic.add_word(x), target)) #ids
            assert len(target) <= max_gloss_len
            target = [word_dic.get_id('<bos>')] + target + [word_dic.get_id('<eos>')]
            target.extend([word_dic.get_id('<pad>')]*(max_gloss_len + 2 - len(target)))
            target_ids.append(target)
         
        self.char_dic = char_dic
        self.word_dic = word_dic
        self.source_ids = torch.LongTensor(source_ids)
        self.target_ids = torch.LongTensor(target_ids)
        
    def __getitem__(self, index):
        return self.source_ids[index], self.target_ids[index]
    
    def __len__(self):
        return len(self.source_ids)
    
class seq2seq(nn.Module):
    def __init__(self, embedding_path, char_dic, word_dic):
        super(seq2seq, self).__init__()
        
        word_emb_dim = 100
        char_emb_dim = 100
        encoder_dim = 512
        decoder_dim = 512
        
        f=open(embedding_path, encoding="utf-8")
        line=f.readline()
        word2emb={}
        while line:
            line=line.split()
            word2emb[line[0]]=torch.from_numpy(np.array(line[1:],dtype=np.str).astype(np.float))
            line=f.readline()
            
        word_emb = nn.Embedding(word_dic.get_len(), word_emb_dim, padding_idx=word_dic.get_id('<pad>'))
        for i in range(word_dic.get_len()):
            word = word_dic.get_word(i).lower()
            if word in word2emb:
                word_emb.weight.data[i] = word2emb[word]
                
        char_emb = nn.Embedding(char_dic.get_len(), char_emb_dim, padding_idx=char_dic.get_id('<pad>'))
        print(char_emb.weight.requires_grad)
        
        self.char_dic = char_dic
        self.word_dic = word_dic
        self.char_emb = char_emb
        self.encoder = nn.LSTM(char_emb_dim, encoder_dim, batch_first=True)
        self.word_emb = word_emb
        self.decoder = nn.LSTM(word_emb_dim, decoder_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(decoder_dim, decoder_dim*2),
                                  nn.ReLU(),
                                  nn.Linear(decoder_dim*2, word_dic.get_len()) )
        
    def forward(self, source, target_input):
        source = self.char_emb(source)
        source_feature, source_states = self.encoder(source)
        target_input = self.word_emb(target_input)
        target_feature, target_states = self.decoder(target_input, source_states)
        output = self.head(target_feature)
        
        return output
    
    def decode(self, source):
        batch_size = source.size(0)
        source = self.char_emb(source)
        source_feature, source_states = self.encoder(source)
        target_ids = torch.full((batch_size, 1), self.word_dic.get_id('<bos>'), dtype=torch.long, device=source.device)
        target_states = source_states
        for i in range(100):
            target_feature = self.word_emb(target_ids[:,-1:])
            target_feature, target_states = self.decoder(target_feature, target_states)
            target_feature = self.head(target_feature).squeeze()
            target_ids = torch.cat( (target_ids, torch.argmax(target_feature, dim=1, keepdim=True)), dim=1)
            if (target_ids.eq(self.word_dic.get_id('<eos>')).sum(dim=1) >= 1).sum().item() == batch_size:
                break
                
        return target_ids
    
    def get_feature(self, source, target_input):
        source = self.char_emb(source)
        source_feature, source_states = self.encoder(source)
        target_input = self.word_emb(target_input)
        target_feature, target_states = self.decoder(target_input, source_states)
        
        batch_size=source_feature.size(0)
        print(source_feature.size(),target_feature.size())
        
        return source_feature, target_feature
    
def id2words(ids, dictionary, ref=False):
    bos = dictionary.get_id('<bos>')
    eos = dictionary.get_id('<eos>')
    
    ids = ids.tolist()
    sents = []
    for sent in ids:
        begin = False
        words = []
        for token_id in sent:
            if token_id == bos:
                begin = True
            elif token_id == eos:
                break
            elif begin:
                words.append(dictionary.get_word(token_id))
        if ref:
            sents.append([words])
        else:
            sents.append(words)
        
    return sents
            
    
data_path = './text/train_100'
embedding_path = './text/glove.6B.100d.txt'
epoch_num = 50
batch_size = 32
lr = 1e-3

wiki_data = wiki_dataset(data_path)
data_loader = DataLoader(wiki_data, batch_size=batch_size, shuffle=True, num_workers=0)
model = seq2seq(embedding_path, wiki_data.char_dic, wiki_data.word_dic)
optimizer = torch.optim.Adam(model.parameters(), lr)
loss_func = nn.CrossEntropyLoss(ignore_index=wiki_data.word_dic.get_id('<pad>'))

best_score = 0.0
for epoch in range(epoch_num):
    model.train()
    loss_lis = []
    for i, (source_ids, target_ids) in enumerate(data_loader):
        target_input = target_ids[:, :-1].contiguous()
        target_label = target_ids[:, 1:].contiguous()
        output = model(source_ids, target_input)
        label_num = output.size(-1)
        loss = loss_func(output.view(-1, label_num), target_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_lis.append(loss.item())
        
    mean_loss = torch.mean(torch.tensor(loss_lis)).item()
    print('epoch %d,  mean loss %.4f,  '%(epoch, mean_loss))
        
    if (epoch+1) % 5 == 0:
        model.eval()
        references = []
        hypotheses = []
        for i, (source_ids, target_ref) in enumerate(data_loader):
            target_hypo = model.decode(source_ids)
            hypotheses.extend(id2words(target_hypo, wiki_data.word_dic))
            references.extend(id2words(target_ref, wiki_data.word_dic, ref=True))
        score = corpus_bleu(references, hypotheses)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), './seq2seq.pt')           
        print('bleu score %.2f'%(score*100))
        print('reference: ' + ' '.join(references[0][0]))
        print('predict: ' + ' '.join(hypotheses[0]))
            
model.load_state_dict(torch.load('./seq2seq.pt'))
source_features = []
target_features = []
for i, (source_ids, target_ids) in enumerate(data_loader):    
            source_feature, target_feature = model.get_feature(source_ids, target_ids[:, :-1].contiguous())
            source_features.append(source_feature)
            target_features.append(target_feature)
source_features = torch.cat(source_features, dim=0)
target_features = torch.cat(target_features, dim=0)

def transform(tensor):
    num, seq_len, hidden_size = tensor.size()
    mean = tensor.view(-1, hidden_size).mean(dim=0)
    tensor = torch.where(tensor > mean, torch.tensor([1,],device=tensor.device), torch.tensor([-1,],device=tensor.device))
    return tensor.view(num, -1), mean

source_features, s_mean = transform(source_features)
target_features, t_mean = transform(target_features)

torch.save((source_features,target_features, s_mean), './tensor.pt')