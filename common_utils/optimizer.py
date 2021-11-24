import torch
from torch.optim import Optimizer
from collections import defaultdict
from transformers import get_linear_schedule_with_warmup


# ================================================================== #
#                    定义优化器和学习策略                                #
# ================================================================== #
def build_optimizer(args, model, total_steps):
    bert_param_optimizer, linear_param_optimizer, prompt_param_optimizer = [], [], []
    for named_param in list(model.named_parameters()):
        if 'bert' in named_param[0]:
            bert_param_optimizer.append(named_param)
        elif 'prompt_embedding' in named_param[0]:
            prompt_param_optimizer.append(named_param)
        else:
            linear_param_optimizer.append(named_param)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # bert
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay, "lr": args.bert_lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0, "lr": args.bert_lr},
        # linear layer
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay, "lr": args.downstream_lr},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0, "lr": args.downstream_lr},
        # prompt embedding layer
        {'params': [p for n, p in prompt_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay, "lr": args.downstream_lr},
        {'params': [p for n, p in prompt_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0, "lr": args.downstream_lr},
    ]
    if args.lookahead:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.bert_lr,
                                      eps=args.adam_epsilon)
        optimizer = Lookahead(optimizer=optimizer, k=args.lookahead_k, alpha=args.lookahead_alpha)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.bert_lr,
                                      eps=args.adam_epsilon)

    scheduler = None
    if args.warmup:
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

    return optimizer, scheduler


class Lookahead(Optimizer):
    '''
    PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''
    def __init__(self, optimizer,alpha=0.5, k=6,pullback_momentum="none"):
        '''
        :param optimizer:inner optimizer
        :param k (int): number of lookahead steps
        :param alpha(float): linear interpolation factor. 1.0 recovers the inner optimizer.
        :param pullback_momentum (str): change to inner optimizer momentum on interpolation update
        '''
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'k':self.k,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(1.0 - self.alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.alpha).add_(
                            1.0 - self.alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
