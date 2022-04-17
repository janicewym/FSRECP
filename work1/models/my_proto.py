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

class MYProto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.drop2 = nn.Dropout(0.2)
        self.dot = dot
        self.hidden_size = 768
       
    def __dist__(self, x, y, dim, dot):
        if dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, dot=False):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3, dot)
    
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
        support_emb,_ = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_emb,_ = self.sentence_encoder(query) # (B * total_Q, D)
        
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