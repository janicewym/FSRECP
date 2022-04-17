from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_prompt_loader, get_kgtypeprompt_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder, \
                                            PromptSentenceEncoder, KGTypePromptSentenceEncoder
import models
from models.my_proto import MYProto
from models.my_kgproto import MYKGTProto
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_type_max_num(types_dict):
    max_len = 0
    types_set = set()
    for types in types_dict.values():
        if max_len < len(types):
            max_len = len(types)
        for t in types:
            t = t.split('/')[-1]
            types_set.add(t)
    return max_len, types_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_semeval',
            help='val file')
    parser.add_argument('--test', default='test_pubmed',
            help='test file')
    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=30000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=-1, type=float,
           help='learning rate')
    parser.add_argument('--proto_lr', default=0.01, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
           help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adamw',
           help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
           help='checkpoint name.')
    parser.add_argument('--ispubmed', default=False, type=bool,
                       help='FewRel 2.0 or not')
    parser.add_argument('--pid2name', default='pid2name',
                        help='pid2name file: relation names and description')
    parser.add_argument('--root', default='./data',
                        help='file root')


    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
           help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
           help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
           help='concatenate entity representation as sentence rep')
    parser.add_argument('--version', default=None,
           help='the version of experiment')
    parser.add_argument('--prompt', action='store_true',
           help='use prompt model')
    parser.add_argument('--kg', action='store_true',
           help='use kg model')
    parser.add_argument('--type', action='store_true',
           help='use entity type')
    parser.add_argument('--con', action='store_true',
           help='use contrastive train')
    
    # only for prototypical networks
    parser.add_argument('--dot', action='store_true', 
           help='use dot instead of L2 distance for proto')
    
    # experiment
    parser.add_argument('--mask_entity', action='store_true',
           help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true',
           help='use SGD instead of AdamW for BERT.')

    parser.add_argument('--seed', type=int, default=0,
           help='random seed')

    opt = parser.parse_args()
    print(vars(opt))
    set_seed(opt.seed)
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    special_tokens_dict = {'additional_special_tokens': ['E1', '/E1', 'E2', '/E2']}

    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length)
    elif encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep,
                    mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'
        if opt.pair:
            sentence_encoder = RobertaPAIRSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    cat_entity_rep=opt.cat_entity_rep)
    elif encoder_name == 'prompt':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        sentence_encoder = PromptSentenceEncoder(
                    pretrain_ckpt,
                    max_length, 
                    special_tokens_dict)
    elif encoder_name == 'kgtypeprompt':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        sentence_encoder = KGTypePromptSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    special_tokens_dict)
    else:
        raise NotImplementedError
    
    if opt.pair:
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
    
    elif opt.prompt and opt.kg is False and opt.type is False:
        train_data_loader = get_prompt_loader(opt.train, opt.pid2name, sentence_encoder,
                N=trainN, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        val_data_loader = get_prompt_loader(opt.val, opt.pid2name, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        test_data_loader = get_prompt_loader(opt.test, opt.pid2name, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
    
    elif opt.prompt and opt.kg and opt.type:
        print("Loading entity dict...")
        wikilist = [one.strip("\n") for one in open('../../KGPrompt_v4_1/data/wiki.txt')]
        entity2id = dict([(wiki, k+1) for k, wiki in enumerate(wikilist)])
        id2entity = dict([(k+1, wiki) for k, wiki in enumerate(wikilist)])
        
        print("Loading relation dict...")
        rellist = [rel.strip("\n") for rel in open('../../KGPrompt_v4_1/data/relations.txt')]
        rel2id = dict([(rel, k) for k, rel in enumerate(rellist)]) # id 对应行下标为id的embedding
        id2rel = dict([(k, rel) for k, rel in enumerate(rellist)])
        
        print("Loading entity type dict...")
        entity2type = json.load(open('../../KGPrompt_v4_1/data/prior/wiki_type_wiki80.json'))
        type_max_num, types_set = get_type_max_num(entity2type)
        
        train_data_loader = get_kgtypeprompt_loader(opt.train, opt.pid2name, entity2id, rel2id, entity2type, type_max_num, sentence_encoder,
                N=trainN, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        val_data_loader = get_kgtypeprompt_loader(opt.val, opt.pid2name, entity2id, rel2id, entity2type, type_max_num, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        test_data_loader = get_kgtypeprompt_loader(opt.test, opt.pid2name, entity2id, rel2id, entity2type, type_max_num, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size, ispubmed=opt.ispubmed, root=opt.root)
        
    else:
        train_data_loader = get_loader(opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
   
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError

    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, opt.version, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    
    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=opt.dot)
    elif model_name == "my_proto":
        model = MYProto(sentence_encoder, dot=opt.dot)
    elif model_name == "my_kgproto":
        model = MYKGTProto(sentence_encoder, id2entity, id2rel, dot=opt.dot)
    else:
        raise NotImplementedError
    
    if not os.path.exists('checkpoint_check'):
        os.mkdir('checkpoint_check')
    ckpt = 'checkpoint_check/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta', 'prompt', 'kgprompt', 'kgtypeprompt']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1
        
        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, 
                learning_rate=opt.lr, proto_lr=opt.proto_lr, use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter, 
                iskg=opt.kg, istype=opt.type, iscontra=opt.con)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair, iskg=opt.kg, istype=opt.type, iscontra=opt.con)
    print("RESULT: %.2f" % (acc * 100))
    
if __name__ == "__main__":
    main()
