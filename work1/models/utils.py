import torch

def total_get_prior(head_id, tail_id, id2rel, score_head_relation_dict, score_tail_relation_dict):
    total_score = 0.0
    for rel in id2rel.values():
        head_rel = head_id + "_" + rel
        tail_rel = tail_id + "_" + rel
        head_score = 0.0
        tail_score = 0.0
        
        if head_rel in score_head_relation_dict:
            head_score = score_head_relation_dict[head_rel]
        
        if tail_rel in score_tail_relation_dict:
            tail_score = score_tail_relation_dict[tail_rel]
            
        total_score += head_score * tail_score
    
    return total_score

def get_one_prior(head_id, tail_id, rel_id, score_head_relation_dict, score_tail_relation_dict):
    head_rel = head_id + "_" + rel_id
    tail_rel = tail_id + "_" + rel_id
    head_score = 0.0
    tail_score = 0.0
    if head_rel in score_head_relation_dict:
        head_score = score_head_relation_dict[head_rel]
    if tail_rel in score_tail_relation_dict:
        tail_score = score_tail_relation_dict[tail_rel]
        
    prior = head_score * tail_score 
    return prior

def get_prior(head_id, tail_id, rel_id, id2entity, id2rel, score_head, score_tail):
    total_num = head_id.size(0)
    priors = list()
    for i in range(total_num):
        # print(head_id[i], tail_id[i])
        h_id = id2entity[head_id[i].item()]
        t_id = id2entity[tail_id[i].item()]
        r_id = id2rel[rel_id[i].item()]
        # 分母
        d = total_get_prior(h_id, t_id, id2rel, score_head, score_tail)
        # 分子
        m = get_one_prior(h_id, t_id, r_id, score_head, score_tail)
        
        if d == 0.0 or m == 0.0:
            priors.append(1.0)
        else:
            priors.append(m / d)
    
    priors = torch.tensor(priors)
    if torch.cuda.is_available():
        priors = priors.cuda()
    return priors
        
