import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

from models.utils import *

"""
原型：mean + relation
loss: 原型 + KE
"""
    
class Combine(nn.Module):
    def __init__(self, hidden_size):
        super(Combine, self).__init__()
        self.sent_in_dim = 2 * hidden_size
        self.linear_combine_relation = nn.Linear(self.sent_in_dim, hidden_size) #, bias=True)

    def forward(self, x):
        x1 = self.linear_combine_relation(x)  # (B * N * K,max_len, H)
        return x1
    
class EntityEnhancement(nn.Module):
    def __init__(self, hidden_size):
        super(EntityEnhancement, self).__init__()
        self.entity_in_dim = 2 * hidden_size
        self.linear_combine_type = nn.Linear(self.entity_in_dim, hidden_size) #, bias=True)

    def forward(self, x):
        x1 = self.linear_combine_type(x)  # (B * N * K,max_len, H)
        return x1

class MYKGTProto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, id2entity, id2rel, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.drop2 = nn.Dropout(0.2)
        self.dot = dot
        self.id2entity = id2entity
        self.id2rel = id2rel
        self.hidden_size = 768
        self.combine = Combine(self.hidden_size)
        self.temp_proto = 1
       
    def __dist__(self, x, y, dim, dot):
        if dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, dot=False):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3, dot)
    
    def __kg_score__(self, h, r, t):
        score = (h + r) - t
        score = torch.exp(-torch.norm(score, p=2, dim=-1))
        return score

    def __get_type_attention__(self, h_state, t_state, htype_embs, ttype_embs, h_nums, t_nums):
        B = h_state.shape[0]
        hidden_size = h_state.shape[-1]
        htype_embs = htype_embs.view(B, -1, hidden_size)
        ttype_embs = ttype_embs.view(B, -1, hidden_size)
        # print(h_state.shape, t_state.shape, htype_embs.shape, ttype_embs.shape)
        allhtype_embs = None
        allttype_embs = None
        for b in range(B):
            hnum = h_nums[b].item()
            tnum = t_nums[b].item()
            
            one_htype_embs = htype_embs[b,:hnum] # (T, D)
            one_ttype_embs = ttype_embs[b,:tnum] # (T, D)
            
            one_hsim = torch.mm(h_state[b].unsqueeze(0), one_htype_embs.T) # (1, T)
            one_halphas = F.softmax(one_hsim, dim=1)  
            one_htype_embs = torch.sum(one_halphas.T * one_htype_embs, dim=0) # (D)
            
            one_tsim = torch.mm(t_state[b].unsqueeze(0), one_ttype_embs.T) # (1, T)
            one_talphas = F.softmax(one_tsim, dim=1)  
            one_ttype_embs = torch.sum(one_talphas.T * one_ttype_embs, dim=0) # (D)
            
            if allhtype_embs is not None:
                allhtype_embs = torch.cat((allhtype_embs, one_htype_embs), 0)
                allttype_embs = torch.cat((allttype_embs, one_ttype_embs), 0)
            else:
                allhtype_embs = one_htype_embs
                allttype_embs = one_ttype_embs
        
        allhtype_embs = allhtype_embs.view(-1, hidden_size) # (B * N * K, D)
        allttype_embs = allttype_embs.view(-1, hidden_size)
        
        return allhtype_embs, allttype_embs

    def __get_type_attention2__(self, state, htype_embs, ttype_embs, h_nums, t_nums):
        B = state.shape[0]
        hidden_size = state.shape[-1]
        htype_embs = htype_embs.view(B, -1, hidden_size)
        ttype_embs = ttype_embs.view(B, -1, hidden_size)
        # print(h_state.shape, t_state.shape, htype_embs.shape, ttype_embs.shape)
        allhtype_embs = None
        allttype_embs = None
        for b in range(B):
            hnum = h_nums[b].item()
            tnum = t_nums[b].item()
            
            one_htype_embs = htype_embs[b,:hnum] # (T, D)
            one_ttype_embs = ttype_embs[b,:tnum] # (T, D)
            
            one_hsim = torch.mm(state[b].unsqueeze(0), one_htype_embs.T) # (1, T)
            one_halphas = F.softmax(one_hsim, dim=1) 
            # one_halphas = one_halphas * (one_halphas > 0.5) # 设置阈值
            one_htype_embs = torch.sum(one_halphas.T * one_htype_embs, dim=0) # (D)
            
            one_tsim = torch.mm(state[b].unsqueeze(0), one_ttype_embs.T) # (1, T)
            one_talphas = F.softmax(one_tsim, dim=1)  
            one_ttype_embs = torch.sum(one_talphas.T * one_ttype_embs, dim=0) # (D)
            
            if allhtype_embs is not None:
                allhtype_embs = torch.cat((allhtype_embs, one_htype_embs), 0)
                allttype_embs = torch.cat((allttype_embs, one_ttype_embs), 0)
            else:
                allhtype_embs = one_htype_embs
                allttype_embs = one_ttype_embs
        
        allhtype_embs = allhtype_embs.view(-1, hidden_size) # (B * N * K, D)
        allttype_embs = allttype_embs.view(-1, hidden_size)
        
        return allhtype_embs, allttype_embs

    
    def forward(self, support, query, N, K, total_Q, eval=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # print("**")
        support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query) # (B * total_Q, D)
        
        # Add relation
        hidden_size = support_emb.size(-1)
        Q = int(total_Q / N)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        
        # Prototypical Networks 
        # Ignore NA policy
      
        support = support.view(-1, N, K, hidden_size) # (B * N * K, 2D) --> (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size) # (B * total_Q, 2D) --> (B, total_Q, D)
        proto = torch.mean(support, 2)  # (B, N, D)
        
        logits = self.__batch_dist__(proto, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B,,total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        return logits, pred