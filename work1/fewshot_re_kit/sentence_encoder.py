import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from . import network
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50, 
            pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, 
                word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, 
                pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

    def forward(self, inputs):
        if not self.cat_entity_rep:
            x = self.bert(inputs['word'], attention_mask=inputs['mask']).pooler_output
            return x
            # x = torch.zeros((inputs['word'].size(0), 768)).type(torch.FloatTensor).cuda()
            # print(x.device)
            # print('3:', torch.cuda.memory_allocated(0))
            # batchsize = 16
            # for i in range(int(inputs['word'].size(0)/batchsize)+1):
            #     print(i)
            #     print('4:', torch.cuda.memory_allocated(0))
            #     print('4 gpu2:', torch.cuda.memory_allocated(1))
            #     x_tmp = self.bert(inputs['word'][i*batchsize:(i+1)*batchsize], attention_mask=inputs['mask'][i*batchsize:(i+1)*batchsize]).pooler_output
            #     # x.append(x_tmp)
            #     print(x_tmp.device)
            #     x[i*batchsize:(i+1)*batchsize] = x_tmp
            #     torch.cuda.empty_cache()
            # return x
        else:
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask

class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens

class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False): 
        nn.Module.__init__(self)
        self.roberta = RobertaModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.cat_entity_rep = cat_entity_rep

    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.roberta(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state

    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        sst = ['<s>'] + sst
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(1)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index, pos2_in_index, mask


class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def forward(self, inputs):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens 


# my sentence encoder
class PromptSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, special_tokens_dict): 
        nn.Module.__init__(self)
        self.bert = BertForMaskedLM.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length_name = 8

        # add specical tokens
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs, cat=False):
        results = self.bert(inputs['word'], attention_mask=inputs['mask'], return_dict=True, output_hidden_states=True)
        outputs = results.hidden_states[-1]
        tensor_range = torch.arange(inputs['word'].size()[0])
        state = outputs[tensor_range, inputs['rel_mask']]
        if not cat:
            return state, outputs
        else:
            h_state = outputs[tensor_range, inputs['pos1']]
            t_state = outputs[tensor_range, inputs['pos2']]
            state = torch.cat((h_state, t_state), -1)
            return state, outputs
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        rel_in_index = 1
        for token in raw_tokens:
            if token != '[MASK]':
                token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[E1]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[E2]')
                pos2_in_index = len(tokens)
            
            tokens += self.tokenizer.tokenize(token)
            
            if cur_pos == pos_head[-1]:
                tokens.append('[/E1]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[/E2]')
            
            if token == '[MASK]':
                rel_in_index = len(tokens)
                
            cur_pos += 1
        tokens += ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        rel_in_index = min(self.max_length, rel_in_index)
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, rel_in_index-1, mask

class KGPromptSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.bert = BertForMaskedLM.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.max_length_name = 8

    def forward(self, inputs, cat=False, isrel=False):
        results = self.bert(inputs['word'], attention_mask=inputs['mask'], return_dict=True, output_hidden_states=True)
        outputs = results.hidden_states[-1]
        tensor_range = torch.arange(inputs['word'].size()[0])
        state = outputs[tensor_range, inputs['rel_mask']]
        
        if isrel:
            return state, outputs
        
        else:
            h_state = outputs[tensor_range, inputs['pos1']]
            t_state = outputs[tensor_range, inputs['pos2']]
            h_id = inputs['head_id']
            t_id = inputs['tail_id']
            r_id = inputs['rel_id']
            if cat:
                state = torch.cat((h_state, t_state), -1)
            return state, outputs, h_state, t_state, h_id, t_id, r_id
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        rel_in_index = 1
        for token in raw_tokens:
            if token != '[MASK]':
                token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[E1]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[E2]')
                pos2_in_index = len(tokens)
            
            tokens += self.tokenizer.tokenize(token)
            
            if cur_pos == pos_head[-1]:
                tokens.append('[/E1]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[/E2]')
            
            if token == '[MASK]':
                rel_in_index = len(tokens)
                
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        rel_in_index = min(self.max_length, rel_in_index)
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, rel_in_index-1, mask

    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

class KGTypePromptSentenceEncoder(nn.Module):
    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False): 
        nn.Module.__init__(self)
        self.bert = BertForMaskedLM.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.max_length_name = 8
        self.max_length_type = 6
        
    def forward(self, inputs, cat=False, isrel=False, istype=False):
        if istype:
            hinputs_word = inputs['head_word']
            hinputs_mask = inputs['head_mask']
            tinputs_word = inputs['tail_word']
            tinputs_mask = inputs['tail_mask']
            
            hinputs_word = hinputs_word.view(-1, hinputs_word.shape[-1])
            hinputs_mask = hinputs_mask.view(-1, hinputs_mask.shape[-1])
            tinputs_word = tinputs_word.view(-1, tinputs_word.shape[-1])
            tinputs_mask = tinputs_mask.view(-1, tinputs_mask.shape[-1]) # (B*T, H) : (200, 6)
            
            hinputs_word = hinputs_word * (hinputs_word > 0)
            hinputs_mask = hinputs_mask * (hinputs_mask >= 0)
            tinputs_word = tinputs_word * (tinputs_word > 0)
            tinputs_mask = tinputs_mask * (tinputs_mask >= 0)
            htype_results = self.bert(hinputs_word, attention_mask=hinputs_mask, return_dict=True, output_hidden_states=True)
            ttype_results = self.bert(tinputs_word, attention_mask=tinputs_mask, return_dict=True, output_hidden_states=True)
            htype_outputs = htype_results.hidden_states[-1] 
            ttype_outputs = ttype_results.hidden_states[-1]
            hstate, tstate = htype_outputs[:, 0], ttype_outputs[:, 0] # (B * T, H)
            
            h_num = inputs['head_num']
            t_num = inputs['tail_num']
            return hstate, tstate, h_num, t_num
        
        results = self.bert(inputs['word'], attention_mask=inputs['mask'], return_dict=True, output_hidden_states=True)
        outputs = results.hidden_states[-1]
        tensor_range = torch.arange(inputs['word'].size()[0])
        state = outputs[tensor_range, inputs['rel_mask']]
        
        if isrel:
            return state, outputs
        
        else:
            h_state = outputs[tensor_range, inputs['pos1']]
            t_state = outputs[tensor_range, inputs['pos2']]
            h_id = inputs['head_id']
            t_id = inputs['tail_id']
            r_id = inputs['rel_id']
            if cat:
                state = torch.cat((state, h_state, t_state), -1)
            return state, outputs, h_state, t_state, h_id, t_id, r_id
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        rel_in_index = 1
        for token in raw_tokens:
            if token != '[MASK]' and token != '[SEP]':
                token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            
            tokens += self.tokenizer.tokenize(token)
            
            # if cur_pos == pos_head[-1]:
                # tokens.append('[/E1]')
            # if cur_pos == pos_tail[-1]:
                # tokens.append('[/E2]')
            
            if token == '[MASK]':
                rel_in_index = len(tokens)
                
            cur_pos += 1
        tokens += ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # print(tokens)
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        rel_in_index = min(self.max_length, rel_in_index)
        # print(tokens)
        # print(indexed_tokens[pos1_in_index - 1])
        # print(indexed_tokens[pos2_in_index - 1])
        # print(self.tokenizer.convert_ids_to_tokens(indexed_tokens[rel_in_index - 1]))
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, rel_in_index-1, mask

    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append(':')
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens += ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens += ['[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_type(self, types):
        # token -> index
        index_tokens_list = []
        mask_list = []
        for one_type in types:
            tokens = ['[CLS]']
            t = one_type.split('/')[-1]
            for token in t.split('_'):
                token = token.lower()
                tokens += self.tokenizer.tokenize(token)
            tokens += ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length_type:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length_type]

            # mask
            mask = np.zeros(self.max_length_type, dtype=np.int32)
            mask[:len(tokens)] = 1
            
            index_tokens_list.append(indexed_tokens)
            mask_list.append(mask)
            
        return index_tokens_list, mask_list