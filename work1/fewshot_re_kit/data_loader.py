import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import pickle
from fewshot_re_kit.template import *

class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print(f"[ERROR] Data file {path} does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask, label):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        # d['label'].append(label)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        label2classname = {}
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, i)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, i)
                count += 1

            query_label += [i] * self.Q

            label2classname[i] = class_name

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask, self.N)
        query_label += [self.N] * Q_na
        
        return support_set, query_set, query_label #, label2classname
    
    def __len__(self):
        return 1000000000

def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    # support_sets, query_sets, query_labels, label2classname = zip(*data)
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    # label2classname = list(label2classname)
    return batch_support, batch_query, batch_label #, label2classname

def get_loader(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,  
            self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word  = self.__getraw__(
                        self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word = self.__getraw__(
                    self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000

def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set
    
    def __len__(self):
        return 1000000000

def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support

def get_loader_unsupervised(name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

# add my loade
# 以Prompt为框架的数据加载
class PromptDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, pid2name, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file {}, {} does not exist!".format(path, pid2name_path))
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        template, kg_head, kg_tail = get_template(item)
        word, pos1, pos2, rel_mask, mask = self.encoder.tokenize(template, kg_head, kg_tail)
        return word, pos1, pos2, rel_mask, mask 
    
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask
    
    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __additem__(self, d, word, pos1, pos2, rel_mask, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['rel_mask'].append(rel_mask)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': [], 'rel_mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [] }
        query_label = []

        for i, class_name in enumerate(target_classes):
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            rel_text_rel_mask = torch.tensor(0).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            relation_set['rel_mask'].append(rel_text_rel_mask)
            
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, rel_mask, mask = self.__getraw__(
                        self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                rel_mask = torch.tensor(rel_mask).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, rel_mask, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, rel_mask, mask)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, query_label, relation_set
    
    def __len__(self):
        return 1000000000

def prompt_collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': []}
    batch_relation = {'word': [], 'mask': [], 'rel_mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation

def get_prompt_loader(name, pid2name, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=prompt_collate_fn, ispubmed=False, root='./data'):
    dataset = PromptDataset(name, pid2name, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

# 以Prompt为框架，增加KG的数据加载
class KGPromptDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, pid2name, entity2id, rel2id, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file {}, {} does not exist!".format(path, pid2name_path))
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.entity2id = entity2id
        self.rel2id = rel2id
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        template, kg_head, kg_tail = get_template(item)
        word, pos1, pos2, rel_mask, mask = self.encoder.tokenize(template, kg_head, kg_tail)
        head_id = self.entity2id[item['h'][1]]
        tail_id = self.entity2id[item['t'][1]]
        return word, pos1, pos2, rel_mask, mask, head_id, tail_id 
    
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask
    
    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __additem__(self, d, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['rel_mask'].append(rel_mask)
        d['mask'].append(mask)
        d['head_id'].append(head_id)
        d['tail_id'].append(tail_id)
        d['rel_id'].append(rel_id)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': [], 'rel_mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            rel_text_rel_mask = torch.tensor(0).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            relation_set['rel_mask'].append(rel_text_rel_mask)
            
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, rel_mask, mask, head_id, tail_id = self.__getraw__(
                        self.json_data[class_name][j])
                rel_id = self.rel2id[class_name]
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                rel_mask = torch.tensor(rel_mask).long()
                mask = torch.tensor(mask).long()
                head_id = torch.tensor(head_id, dtype=torch.int)
                tail_id = torch.tensor(tail_id, dtype=torch.int)
                rel_id = torch.tensor(rel_id, dtype=torch.int)
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id)
                else:
                    self.__additem__(query_set, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id)
                count += 1

            query_label += [i] * self.Q

        return support_set, query_set, query_label, relation_set
    
    def __len__(self):
        return 1000000000

def kgprompt_collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
    batch_relation = {'word': [], 'mask': [], 'rel_mask': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation

def get_kgprompt_loader(name, pid2name, entity2id, rel2id, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=kgprompt_collate_fn, ispubmed=False, root='./data'):
    dataset = KGPromptDataset(name, pid2name, entity2id, rel2id, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)

# 以Prompt为框架，增加KG-type的数据加载
class KGTypePromptDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file {}, {} does not exist!".format(path, pid2name_path))
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.entity2id = entity2id
        self.rel2id = rel2id
        self.entity2type = entity2type
        self.type_max_num = type_max_num
        self.max_length_type = 6
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        template, kg_head, kg_tail = get_template(item)
        word, pos1, pos2, rel_mask, mask = self.encoder.tokenize(template, kg_head, kg_tail)
        head_id = self.entity2id[item['h'][1]]
        tail_id = self.entity2id[item['t'][1]]
        return word, pos1, pos2, rel_mask, mask, head_id, tail_id 
    
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask
    
    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __gettype__(self, types, word, mask):
        cword, cmask = self.encoder.tokenize_type(types)
        for idx in range(len(cword)):
            word[idx] = torch.tensor(cword[idx])
            mask[idx] = torch.tensor(cmask[idx])
        return word, mask, len(cword)
    
    def __additem__(self, d, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['rel_mask'].append(rel_mask)
        d['mask'].append(mask)
        d['head_id'].append(head_id)
        d['tail_id'].append(tail_id)
        d['rel_id'].append(rel_id)

    def __addtype__(self, d, head_word, head_mask, tail_word, tail_mask, head_num, tail_num):
        d['head_word'].append(head_word)
        d['head_mask'].append(head_mask)
        d['tail_word'].append(tail_word)
        d['tail_mask'].append(tail_mask)
        d['head_num'].append(head_num)
        d['tail_num'].append(tail_num)
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': [], 'rel_mask': []}
        support_type_set = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
        query_type_set = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            # relation
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            rel_text_rel_mask = torch.tensor(0).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            relation_set['rel_mask'].append(rel_text_rel_mask)
            
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, rel_mask, mask, head_id, tail_id = self.__getraw__(
                        self.json_data[class_name][j])
                rel_id = self.rel2id[class_name]
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                rel_mask = torch.tensor(rel_mask).long()
                mask = torch.tensor(mask).long()
                head_id = torch.tensor(head_id, dtype=torch.int)
                tail_id = torch.tensor(tail_id, dtype=torch.int)
                rel_id = torch.tensor(rel_id, dtype=torch.int)
                
                # entity type
                head = self.json_data[class_name][j]['h'][1]
                tail = self.json_data[class_name][j]['t'][1]
                head_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                head_mask = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                tail_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                tail_mask = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                
                head_type_num = 0
                tail_type_num = 0
                
                if head in self.entity2type:
                    head_word, head_mask, head_type_num = self.__gettype__(self.entity2type[head], head_word, head_mask)
                if tail in self.entity2type:
                    tail_word, tail_mask, tail_type_num = self.__gettype__(self.entity2type[tail], tail_word, tail_mask)
                
                head_type_num = torch.tensor(head_type_num).long()
                tail_type_num = torch.tensor(tail_type_num).long()
                
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id)
                    self.__addtype__(support_type_set, head_word, head_mask, tail_word, tail_mask, head_type_num, tail_type_num)
                else:
                    self.__additem__(query_set, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id)
                    self.__addtype__(query_type_set, head_word, head_mask, tail_word, tail_mask, head_type_num, tail_type_num)
                count += 1

            query_label += [i] * self.Q
        return support_set, query_set, query_label, relation_set, support_type_set, query_type_set
    
    def __len__(self):
        return 1000000000

def kgtypeprompt_collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
    batch_relation = {'word': [], 'mask': [], 'rel_mask': []}
    batch_support_type = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
    batch_query_type = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets,  support_type_sets, query_type_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        for k in support_type_sets[i]:
            batch_support_type[k] += support_type_sets[i][k]
        for k in query_type_sets[i]:
            batch_query_type[k] += query_type_sets[i][k]
        batch_label += query_labels[i]
    
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    for k in batch_support_type:
        batch_support_type[k] = torch.stack(batch_support_type[k], 0)
    for k in batch_query_type:
        batch_query_type[k] = torch.stack(batch_query_type[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation, batch_support_type, batch_query_type

def get_kgtypeprompt_loader(name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=kgtypeprompt_collate_fn, ispubmed=False, root='./data'):
    dataset = KGTypePromptDataset(name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


class KGTypePromptDataset2(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file {}, {} does not exist!".format(path, pid2name_path))
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.entity2id = entity2id
        self.rel2id = rel2id
        self.entity2type = entity2type
        self.type_max_num = type_max_num
        self.max_length_type = 6
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item): # mask1: head, mask2: tail, mask3: rel
        mask_token = self.encoder.tokenizer.mask_token
        sep_token = self.encoder.tokenizer.sep_token
        template, mask_pos1, mask_pos2, mask_pos3 = get_template2(item, sep_token, mask_token)
        word, mask1, mask2, mask3, mask = self.encoder.tokenize(template, mask_pos1, mask_pos2, mask_pos3)
        # head_id = self.entity2id[item['h'][1]]
        # tail_id = self.entity2id[item['t'][1]]
        return word, mask1, mask2, mask3, mask
    
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask
    
    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __gettype__(self, types, word, onlyword, mask):
        cword, onlycword, cmask = self.encoder.tokenize_type(types)
        for idx in range(len(cword)):
            word[idx] = torch.tensor(cword[idx])
            mask[idx] = torch.tensor(cmask[idx])
            onlyword[idx] = torch.tensor(onlycword[idx])
        return word, onlyword, mask, len(cword)
    
    def __additem__(self, d, word, mask1, mask2, mask3, mask, head_word, tail_word, head_type_num, tail_type_num):
        d['word'].append(word)
        d['htype_mask'].append(mask1)
        d['ttype_mask'].append(mask2)
        d['rel_mask'].append(mask3)
        d['mask'].append(mask)
        d['htype_word'].append(head_word)
        d['ttype_word'].append(tail_word)
        d['hnum'].append(head_type_num)
        d['tnum'].append(tail_type_num)

    def __addtype__(self, d, head_word, head_mask, tail_word, tail_mask, head_num, tail_num):
        d['head_word'].append(head_word)
        d['head_mask'].append(head_mask)
        d['tail_word'].append(tail_word)
        d['tail_mask'].append(tail_mask)
        d['head_num'].append(head_num)
        d['tail_num'].append(tail_num)
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': [], 'rel_mask': []}
        support_type_set = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
        query_type_set = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
        support_set = {'word': [], 'mask': [], 'htype_mask': [], 'ttype_mask': [], 'rel_mask': [], 'htype_word': [], 'ttype_word': [], 'hnum': [], 'tnum': []}
        query_set = {'word': [], 'mask': [], 'htype_mask': [], 'ttype_mask': [], 'rel_mask': [], 'htype_word': [], 'ttype_word': [], 'hnum': [], 'tnum': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            # relation
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            rel_text_rel_mask = torch.tensor(0).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            relation_set['rel_mask'].append(rel_text_rel_mask)
            
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, mask1, mask2, mask3, mask = self.__getraw__(
                        self.json_data[class_name][j])
                rel_id = self.rel2id[class_name]
                word = torch.tensor(word).long()
                mask1 = torch.tensor(mask1).long()
                mask2 = torch.tensor(mask2).long()
                mask3 = torch.tensor(mask3).long()
                mask = torch.tensor(mask).long()
                
                # entity type
                head = self.json_data[class_name][j]['h'][1]
                tail = self.json_data[class_name][j]['t'][1]
                head_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                only_head_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                head_mask = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                tail_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                only_tail_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                tail_mask = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                
                head_type_num = 0
                tail_type_num = 0
                
                if head in self.entity2type:
                    head_word, only_head_word, head_mask, head_type_num = self.__gettype__(self.entity2type[head], head_word, only_head_word, head_mask)
                if tail in self.entity2type:
                    tail_word, only_tail_word, tail_mask, tail_type_num = self.__gettype__(self.entity2type[tail], tail_word, only_tail_word, tail_mask)
                
                head_type_num = torch.tensor(head_type_num).long()
                tail_type_num = torch.tensor(tail_type_num).long()
                
                if count < self.K:
                    self.__additem__(support_set, word, mask1, mask2, mask3, mask, only_head_word, only_tail_word, head_type_num, tail_type_num)
                    self.__addtype__(support_type_set, head_word, head_mask, tail_word, tail_mask, head_type_num, tail_type_num)
                else:
                    self.__additem__(query_set, word, mask1, mask2, mask3, mask, only_head_word, only_tail_word, head_type_num, tail_type_num)
                    self.__addtype__(query_type_set, head_word, head_mask, tail_word, tail_mask, head_type_num, tail_type_num)
                count += 1

            query_label += [i] * self.Q
        return support_set, query_set, query_label, relation_set, support_type_set, query_type_set
    
    def __len__(self):
        return 1000000000

def kgtypeprompt_collate_fn2(data):
    batch_support = {'word': [], 'mask': [], 'htype_mask': [], 'ttype_mask': [], 'rel_mask': [], 'htype_word': [], 'ttype_word': [], 'hnum': [], 'tnum': []}
    batch_query = {'word': [], 'mask': [], 'htype_mask': [], 'ttype_mask': [], 'rel_mask': [], 'htype_word': [], 'ttype_word': [], 'hnum': [], 'tnum': []}
    batch_relation = {'word': [], 'mask': [], 'rel_mask': []}
    batch_support_type = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
    batch_query_type = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets,  support_type_sets, query_type_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        for k in support_type_sets[i]:
            batch_support_type[k] += support_type_sets[i][k]
        for k in query_type_sets[i]:
            batch_query_type[k] += query_type_sets[i][k]
        batch_label += query_labels[i]
    
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    for k in batch_support_type:
        batch_support_type[k] = torch.stack(batch_support_type[k], 0)
    for k in batch_query_type:
        batch_query_type[k] = torch.stack(batch_query_type[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation, batch_support_type, batch_query_type

def get_kgtypeprompt_loader2(name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=kgtypeprompt_collate_fn2, ispubmed=False, root='./data'):
    dataset = KGTypePromptDataset2(name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)


class KGTypePromptConDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, root, ispubmed=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file {}, {} does not exist!".format(path, pid2name_path))
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.entity2id = entity2id
        self.rel2id = rel2id
        self.entity2type = entity2type
        self.type_max_num = type_max_num
        self.max_length_type = 6
        self.N = N
        self.K = K
        self.Q = Q
        self.ispubmed = ispubmed
        self.encoder = encoder

    def __getraw__(self, item):
        template, kg_head, kg_tail = get_template(item)
        word, pos1, pos2, rel_mask, mask = self.encoder.tokenize(template, kg_head, kg_tail)
        head_id = self.entity2id[item['h'][1]]
        tail_id = self.entity2id[item['t'][1]]
        return word, pos1, pos2, rel_mask, mask, head_id, tail_id 
    
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask
    
    def __getname__(self, name):
        word, mask = self.encoder.tokenize_name(name)
        return word, mask

    def __gettype__(self, types, word, mask):
        cword, cmask = self.encoder.tokenize_type(types)
        for idx in range(len(cword)):
            word[idx] = torch.tensor(cword[idx])
            mask[idx] = torch.tensor(cmask[idx])
        return word, mask, len(cword)
    
    def __additem__(self, d, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['rel_mask'].append(rel_mask)
        d['mask'].append(mask)
        d['head_id'].append(head_id)
        d['tail_id'].append(tail_id)
        d['rel_id'].append(rel_id)

    def __addtype__(self, d, head_word, head_mask, tail_word, tail_mask, head_num, tail_num):
        d['head_word'].append(head_word)
        d['head_mask'].append(head_mask)
        d['tail_word'].append(tail_word)
        d['tail_mask'].append(tail_mask)
        d['head_num'].append(head_num)
        d['tail_num'].append(tail_num)
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        relation_set = {'word': [], 'mask': [], 'rel_mask': []}
        support_type_set = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
        query_type_set = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            # relation
            if self.ispubmed:
                if class_name in self.pid2name.keys():
                    name, _ = self.pid2name[class_name]
                    rel_text, rel_text_mask = self.__getname__(name)
                else:
                    rel_text, rel_text_mask = self.__getname__(class_name)
            else:
                rel_text, rel_text_mask = self.__getrel__(self.pid2name[class_name])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            rel_text_rel_mask = torch.tensor(0).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            relation_set['rel_mask'].append(rel_text_rel_mask)
            
            indices = np.random.choice(
                    list(range(len(self.json_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, rel_mask, mask, head_id, tail_id = self.__getraw__(
                        self.json_data[class_name][j])
                rel_id = self.rel2id[class_name]
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                rel_mask = torch.tensor(rel_mask).long()
                mask = torch.tensor(mask).long()
                head_id = torch.tensor(head_id, dtype=torch.int)
                tail_id = torch.tensor(tail_id, dtype=torch.int)
                rel_id = torch.tensor(rel_id, dtype=torch.int)
                
                # entity type
                head = self.json_data[class_name][j]['h'][1]
                tail = self.json_data[class_name][j]['t'][1]
                head_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                head_mask = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                tail_word = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                tail_mask = torch.ones([self.type_max_num, self.max_length_type], dtype=torch.long) * (-1)
                
                head_type_num = 0
                tail_type_num = 0
                
                if head in self.entity2type:
                    head_word, head_mask, head_type_num = self.__gettype__(self.entity2type[head], head_word, head_mask)
                if tail in self.entity2type:
                    tail_word, tail_mask, tail_type_num = self.__gettype__(self.entity2type[tail], tail_word, tail_mask)
                
                head_type_num = torch.tensor(head_type_num).long()
                tail_type_num = torch.tensor(tail_type_num).long()
                
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id)
                    self.__addtype__(support_type_set, head_word, head_mask, tail_word, tail_mask, head_type_num, tail_type_num)
                else:
                    self.__additem__(query_set, word, pos1, pos2, rel_mask, mask, head_id, tail_id, rel_id)
                    self.__addtype__(query_type_set, head_word, head_mask, tail_word, tail_mask, head_type_num, tail_type_num)
                count += 1

            query_label += [i] * self.Q
        return support_set, query_set, query_label, relation_set, support_type_set, query_type_set
    
    def __len__(self):
        return 1000000000

def kgtypepromptcon_collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'rel_mask': [], 'head_id': [], 'tail_id': [], 'rel_id': []}
    batch_relation = {'word': [], 'mask': [], 'rel_mask': []}
    batch_support_type = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
    batch_query_type = {'head_word': [], 'head_mask': [], 'tail_word': [], 'tail_mask': [], 'head_num': [], 'tail_num': []}
    batch_label = []
    support_sets, query_sets, query_labels, relation_sets,  support_type_sets, query_type_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        for k in support_type_sets[i]:
            batch_support_type[k] += support_type_sets[i][k]
        for k in query_type_sets[i]:
            batch_query_type[k] += query_type_sets[i][k]
        batch_label += query_labels[i]
    
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    for k in batch_support_type:
        batch_support_type[k] = torch.stack(batch_support_type[k], 0)
    for k in batch_query_type:
        batch_query_type[k] = torch.stack(batch_query_type[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_relation, batch_support_type, batch_query_type

def get_kgtypepromptcon_loader(name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, batch_size, 
        num_workers=8, collate_fn=kgtypepromptcon_collate_fn, ispubmed=False, root='./data'):
    dataset = KGTypePromptConDataset(name, pid2name, entity2id, rel2id, entity2type, type_max_num, encoder, N, K, Q, root, ispubmed)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)
