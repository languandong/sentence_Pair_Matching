from pretrain_args import args, model_map
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from common_utils.DataLoaders import BlockShuffleDataLoader
import numpy as np
import torch
from models.pretrain_model import Model
from common_utils.util import load_data, get_save_path, get_logger, set_seed, FGM
from common_utils.MyDataset import BlockShufflePretrainDataset
import os


# 是否使用多GPU
if args.use_multi_gpu:
    torch.distributed.init_process_group(backend="nccl")
    if args.local_rank == 0:
        # 创建模型保存路径以及日志
        get_save_path(args)
        logger = get_logger(args.model_save_path + '/pretrain.log', display2console=False)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 创建模型保存路径以及日志
    get_save_path(args)
    logger = get_logger(args.model_save_path + '/pretrain.log', display2console=False)

# 读取预训练数据
train_data = []
for train_path in args.train_data_path:
    train_data += load_data(train_path)

# 训练函数
def train(model):
    set_seed(args.seed)

    if args.use_multi_gpu:
        pretrain_dataset = BlockShufflePretrainDataset(train_data, args.max_seq_length)
        pretrain_sampler = DistributedSampler(pretrain_dataset)
        train_loader = BlockShuffleDataLoader(pretrain_dataset, batch_size=args.batch_size, sampler=pretrain_sampler)
    else:
        pretrain_dataset = BlockShufflePretrainDataset(train_data, args.max_seq_length)
        train_loader = BlockShuffleDataLoader(pretrain_dataset, batch_size=args.batch_size, is_shuffle=True,
                                          sort_key=lambda x: len(x[0]) + len(x[1]),
                                          collate_fn=pretrain_dataset.collate)

    total_steps = len(train_loader) * ((args.epoch - args.resume_epoch) + 20)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    warmup_steps = int(total_steps * args.warmup_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.use_multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    fgm = None
    if args.attack_type == 'fgm':
        fgm = FGM(model)

    model.zero_grad()

    for epoch in range(args.resume_epoch, args.epoch):
        if args.use_multi_gpu:
            pretrain_sampler.set_epoch(epoch)
        model.train()
        pbar = tqdm(train_loader, ncols=150)
        losses = []
        for step, data in enumerate(pbar):
            optimizer.zero_grad()

            if args.use_multi_gpu:
                inputs = {
                    'input_ids': data['input_ids'].cuda(args.local_rank, non_blocking=True).long(),
                    'attention_mask': data['attention_mask'].cuda(args.local_rank, non_blocking=True).long(),
                    'token_type_ids': data['token_type_ids'].cuda(args.local_rank, non_blocking=True).long(),
                    'labels': data['output_ids'].cuda(args.local_rank, non_blocking=True).long()
                }

            else:
                inputs = {
                    'input_ids': data['input_ids'].to(args.device).long(),
                    'attention_mask': data['attention_mask'].to(args.device).long(),
                    'token_type_ids': data['token_type_ids'].to(args.device).long(),
                    'labels': data['output_ids'].to(args.device).long()
                }

            masked_lm_loss = model(**inputs)[0]

            if args.gradient_accumulation_steps > 1:
                masked_lm_loss = masked_lm_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(masked_lm_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                masked_lm_loss.backward()

            if args.attack_type == 'fgm':
                fgm.attack(args.fgm_epsilon)
                loss_adv = model(**inputs)[0]
                loss_adv.backward()
                fgm.restore()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            losses.append(masked_lm_loss.cpu().detach().numpy())
            pbar.set_description(
                f'epoch:{epoch + 1} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f}')

        if args.local_rank in [0, -1]:
            logger.info(
                f'epoch:{epoch + 1} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f}')

            if (epoch + 1) % 10 == 0:
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                           args.model_save_path + f'{args.model_type}-epoch{epoch + 1}.pth',
                           _use_new_zipfile_serialization=False)
            else:
                torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                           args.model_save_path + f'{args.model_type}.pth',
                           _use_new_zipfile_serialization=False)


def main():
    set_seed(args.seed)

    if args.use_multi_gpu:
        torch.cuda.set_device(args.local_rank)
        if args.do_resume:
            model = Model(args=args,resume=True)
            model.load_state_dict(torch.load(args.resume_model_path))
            model.cuda(args.local_rank)
            train(model)
        else:
            model = Model(args=args)
            model.cuda(args.local_rank)
            train(model)
    else:
        if args.do_resume:
            model = Model(args=args,resume=True)
            model.load_state_dict(torch.load(args.resume_model_path))
            model.to(args.device)
            train(model)
        else:
            model = Model(args=args)
            model.to(args.device)
            train(model)


if __name__ == '__main__':
    main()
