import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        # self.sentence_encoder = my_sentence_encoder
        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))
    
    def ce_loss(self, logits, temperature=0.07):
        '''
        logits: Logits with the size (..., pos, neg1, neg2, ...)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        pos, neg = logits[:, 0], logits[:, 1:]
        pos = pos.unsqueeze(1) 
        
        dtype = neg.dtype
        small_val = torch.finfo(dtype).tiny
        # print(pos, neg)
        max_val = torch.max(
                pos, torch.max(neg, dim=1, keepdim=True)[0]
            ).detach()
        # print(max_val)
        
        numerator = torch.exp(pos - max_val).squeeze(1)
        denominator = torch.sum(torch.exp(neg - max_val), dim=1) + numerator
        log_exp = - (torch.log((numerator / denominator) + small_val))
        return log_exp.sum()

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None, margin=1):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        self.margin = margin
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.kg_loss = nn.MarginRankingLoss(self.margin)
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def _get_optimizer_params(self, model, bert_optim=False):
        parameters_to_optimize = list(model.named_parameters())
        if bert_optim:
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        else:
            no_decay = []
            
        parameters_to_optimize = [
            {
                    'params': [p for n, p in model.sentence_encoder.named_parameters() 
                    if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01
            },
            {
                    'params': [p for n, p in model.sentence_encoder.named_parameters() 
                    if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
            },
            
        ]

        parameters_to_optimize_for_big_protos = []
        if hasattr(model, "proto_param"):
            print('proto param exists')
            parameters_to_optimize.append(
                    {
                        # center_d parameters
                        'params': [p for p in model.proto_param.center_d.parameters()],
                        'weight_decay': 0.01
                    }
                )
            # parameters_to_optimize.append(
            #         {
            #             # center_d parameters
            #             'params': [p for p in model.proto_centers.parameters()],
            #             'weight_decay': 0.01
            #         }
            #     )
            parameters_to_optimize_for_big_protos = [
                    {
                            'params': [p for p in model.proto_param.radius_d.parameters()],
                            'weight_decay': 0.0
                    }
            ]
            # parameters_to_optimize_for_big_protos = [
            #         {
            #                 'params': [model.proto_radii],
            #                 'weight_decay': 0.0
            #         }
            # ]
        return parameters_to_optimize, parameters_to_optimize_for_big_protos

    def _update_params(self, model, optimizer, optimizer_for_big_protos):
        optimizer.param_groups[-1]['params'] = [p for p in model.proto_param.center_d.parameters()]
        optimizer_for_big_protos.param_groups[-1]['params'] = [p for p in model.proto_param.radius_d.parameters()]
        # pass

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              proto_lr=1e-2,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False,
              iskg=False, 
              istype=False,
              iscontra=False,
              lamda=1): # 5.4 lamda=0.7
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
        # Init
        parameters_to_optimize, parameters_to_optimize_for_big_protos = self._get_optimizer_params(model, bert_optim=bert_optim)
        # optimizer, proto_optimizer = self._init_optimizers(model, learning_rate, proto_lr, use_sgd_for_bert=use_sgd_for_bert, bert_optim=bert_optim)
        if bert_optim:
            print('Use bert optim!')
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)

            # if self.adv:
            #     optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(parameters_to_optimize,
                    learning_rate, weight_decay=weight_decay)
            # if self.adv:
            #     optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        
        proto_optimizer = None
        # if hasattr(model, 'proto_param'):
        #    proto_optimizer = optim.Adam(parameters_to_optimize_for_big_protos, lr=proto_lr)

        # if self.adv:
        #     optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            if pair:
                batch, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    label = label.cuda()
                logits, pred = model(batch, N_for_train, K, 
                        Q * N_for_train + na_rate * Q)
            elif iskg and istype is False:
                support, query, label, rel_text = next(self.train_data_loader) # min
                # print(support.keys())
                # print('1:', torch.cuda.memory_allocated(0))
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    label = label.cuda()
                # print('2:', torch.cuda.memory_allocated(0))

                # logits, pred = model(support, query, 
                #        N_for_train, K, Q * N_for_train + na_rate * Q, label2classname)
                logits, pred, logits_proto, labels_proto, q_logits_proto, q_labels_proto = model(support, query, 
                        N_for_train, K, Q * N_for_train + na_rate * Q, rel_text) # min            
            
            elif iskg and istype:
                support, query, label, rel_text, support_type, query_type = next(self.train_data_loader) # min
                    
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    for k in support_type:
                        support_type[k] = support_type[k].cuda()
                    for k in query_type:
                        query_type[k] = query_type[k].cuda()
                    label = label.cuda()
                if iscontra:
                    logits, pred, inter_logits, sintra_logits, qintra_logits = model(support, query, 
                            N_for_train, K, Q * N_for_train + na_rate * Q, rel_text, support_type, query_type) 
                
                else:
                    logits, pred = model(support, query, 
                            N_for_train, K, Q * N_for_train + na_rate * Q, rel_text, support_type, query_type) 
            else:
                # support, query, label, label2classname = next(self.train_data_loader) 
                support, query, label, rel_text = next(self.train_data_loader) # min
                # print(support.keys())
                # print('1:', torch.cuda.memory_allocated(0))
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    label = label.cuda()
                # print('2:', torch.cuda.memory_allocated(0))

                # logits, pred = model(support, query, 
                #        N_for_train, K, Q * N_for_train + na_rate * Q, label2classname)
                logits, pred = model(support, query, 
                        N_for_train, K, Q * N_for_train + na_rate * Q, rel_text) # min
                
            if iskg:
                if iscontra:
                    # loss = model.loss(logits, label) / float(grad_iter) + model.ce_loss(inter_logits) + model.ce_loss(sintra_logits) + model.ce_loss(qintra_logits)
                    # loss = model.loss(logits, label) / float(grad_iter) + model.ce_loss(inter_logits) # onecone
                    # loss = model.loss(logits, label) / float(grad_iter) + 0.02 * model.ce_loss(inter_logits) + 0.015 * model.ce_loss(sintra_logits) # twocon-v2.1
                    loss = model.loss(logits, label) / float(grad_iter) + model.ce_loss(inter_logits) + model.ce_loss(sintra_logits)
                    # loss = model.loss(logits, label) / float(grad_iter) +  model.ce_loss(inter_logits) +  model.ce_loss(sintra_logits)
                else:
                    loss = model.loss(logits, label) / float(grad_iter)

            else:   
                loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)
            
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            
            if it % grad_iter == 0:
                if proto_optimizer:
                    self._update_params(model, optimizer, proto_optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if proto_optimizer:
                    proto_optimizer.step()
                    proto_optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, current best: {3:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample, best_acc * 100) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, 
                        na_rate=na_rate, pair=pair, iskg=iskg, istype=istype, iscontra=iscontra)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None,
            iskg=False,
            istype=False,
            iscontra=False): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                if pair:
                    batch, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                        label = label.cuda()
                    logits, pred = model(batch, N, K, Q * N + Q * na_rate)
                elif iskg and istype is False:
                    support, query, label, rel_text = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        for k in rel_text:
                            rel_text[k] = rel_text[k].cuda()
                        label = label.cuda()
                    # logits, pred = model(support, query, N, K, Q * N + Q * na_rate, label2classname, eval=True) 
                    logits, pred, _, _, _, _ = model(support, query, N, K, Q * N + Q * na_rate, rel_text, eval=True) # min
                    
                elif iskg and istype:
                    support, query, label, rel_text, support_type, query_type = next(eval_dataset)
                    
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        for k in rel_text:
                            rel_text[k] = rel_text[k].cuda()
                        for k in support_type:
                            support_type[k] = support_type[k].cuda()
                        for k in query_type:
                            query_type[k] = query_type[k].cuda()
                        label = label.cuda()
                    if iscontra:
                        logits, pred, _, _, _ = model(support, query, N, K, Q * N + Q * na_rate, rel_text, support_type, query_type, eval=True) 
                    else:
                        logits, pred = model(support, query, N, K, Q * N + Q * na_rate, rel_text, support_type, query_type, eval=True) # min
                else:
                    # support, query, label, label2classname = next(eval_dataset)
                    support, query, label, rel_text = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        for k in rel_text:
                            rel_text[k] = rel_text[k].cuda()
                        label = label.cuda()
                    # logits, pred = model(support, query, N, K, Q * N + Q * na_rate, label2classname, eval=True) 
                    logits, pred = model(support, query, N, K, Q * N + Q * na_rate, rel_text, eval=True) # min

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample
