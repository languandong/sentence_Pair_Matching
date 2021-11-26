import pandas as pd
from finetune_args import args
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from common_utils.optimizer import build_optimizer
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import torch
import torch.nn as nn
import os
from common_utils.util import load_data, get_save_path, get_logger, set_seed, FGM, compute_kl_loss, get_homophonesIdx, EMA, prob_postprocess
from common_utils.MyDataset import FineTuneDataset, BlockShuffleDataset
from common_utils.DataLoaders import BlockShuffleDataLoader
from models.finetune_model import Model

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# 读取训练数据
train_data, valid_data = [], []
for train_path in args.train_data_path:
    train_data += load_data(train_path)

for valid_path in args.valid_data_path:
    valid_data += load_data(valid_path)


# ================================================================== #
#                          训练                                       #
# ================================================================== #
def train(model):
    # 创建模型保存路径以及记录配置的参数
    get_save_path(args)
    # 创建日志对象，写入训练效果
    logger = get_logger(args.model_save_path + '/finetune.log')
    # 创建数据集
    #train_dataset = FineTuneDataset(train_data, args.max_len)
    #valid_dataset = FineTuneDataset(valid_data, args.max_len)
    train_dataset = BlockShuffleDataset(train_data, args.max_len)
    valid_dataset = BlockShuffleDataset(valid_data, args.max_len)

    # 读取数据集
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = BlockShuffleDataLoader(train_dataset, batch_size=args.batch_size, is_shuffle=True,
                                          sort_key=lambda x: len(x[0]) + len(x[1]),
                                          collate_fn=train_dataset.collate)
    valid_loader = BlockShuffleDataLoader(valid_dataset, batch_size=args.batch_size, is_shuffle=False,
                                          sort_key=lambda x: len(x[0]) + len(x[1]),
                                          collate_fn=valid_dataset.collate)
    # 优化器及学习率策略
    optimizer, scheduler = build_optimizer(args, model, total_steps=len(train_loader) * (args.epoch + args.extra_epoch))
    # 损失函数
    criterion = nn.BCELoss()
    # 是否使用对抗生成网络
    fgm = None
    if args.use_fgm:
        fgm = FGM(model)
    model.zero_grad()
    # EMA
    ema = EMA(model, 0.999)
    ema.register()
    # 训练
    best_acc = 0.
    early_stop = 0
    for epoch in range(args.epoch):
        model.train()
        losses, acc_list = [], []
        pbar = tqdm(train_loader, dynamic_ncols=True, desc='训练中')
        # 单个batch的训练过程
        for data in pbar:
            # 梯度是累积计算而不是被替换，因此每个batch将梯度初始化为零
            optimizer.zero_grad()
            # 组织待输入模型的数据，转到GPU上
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long(),
            }
            data['label'] = data['label'].to(args.device).float()
            # 损失函数的定义
            if args.use_rDrop:
                # keep dropout and forward twice
                outputs = model.forward(inputs)
                outputs2 = model.forward(inputs)
                # cross entropy loss for classifier
                ce_loss = 0.5 * (criterion(outputs, data['label']) + criterion(outputs2, data['label']))
                kl_loss = compute_kl_loss(outputs, outputs2)
                # carefully choose hyper-parameters
                loss = ce_loss + args.rDrop_coef * kl_loss
            else:
                outputs = model.forward(inputs)
                loss = criterion(outputs, data['label'])
            # 反向传播
            loss.backward()

            if args.use_fgm:
                fgm.attack(epsilon=args.fgm_epsilon)
                outputs_adv = model(inputs)
                loss_adv = criterion(outputs_adv, data['label'])
                loss_adv.backward()
                fgm.restore()

            optimizer.step()
            if args.warmup:
                scheduler.step()
            ema.update()

            losses.append(loss.cpu().detach().numpy())
            output_array = outputs.cpu().detach().numpy()
            label_array = data['label'].cpu().detach().numpy()
            acc_list.extend(np.argmax(output_array, axis=1) == np.argmax(label_array, axis=1))
            pbar.set_description(
                f'epoch:{epoch + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

        ema.apply_shadow()
        model.eval()
        # 评估验证集
        acc, report = evaluate(model, valid_loader)
        if acc > best_acc:
            early_stop = 0
            best_acc = acc
            if not os.path.exists(args.model_save_path):
                os.mkdir(args.model_save_path)
            # 保存最优效果模型
            torch.save(model.state_dict(), args.model_save_path + f'/{args.model_type}_{args.struc}_best_model.pth',
                       _use_new_zipfile_serialization=False)
        # 效果增长不起时候停掉训练
        else:
            early_stop += 1
            if early_stop > 3:
                break

        logger.info(f'epoch:{epoch + 1}/{args.epoch}, vaild acc: {acc}, best_acc: {best_acc}')
        logger.info(f'{report}')


# ================================================================== #
#                        验证                                         #
# ================================================================== #
def evaluate(model, data_loader):
    true, preds = [], []
    pbar = tqdm(data_loader, dynamic_ncols=True)
    with torch.no_grad():
        for data in pbar:
            data['label'] = data['label'].float()
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long(),
            }
            outputs = model(inputs)
            pred = np.argmax(outputs.cpu().numpy(), axis=-1)
            true.extend(np.argmax(data['label'].cpu().numpy(), axis=-1))
            preds.extend(pred)
    acc = accuracy_score(true, preds)
    report = classification_report(true, preds)

    return acc, report


# ================================================================== #
#                         预测                                        #
# ================================================================== #
def predict(model):
    time_start = time()
    set_seed(args.seed)

    test_data = load_data(args.test_data_path)
    homoIdx = get_homophonesIdx(args.test_data_path)

    test_dataset = FineTuneDataset(test_data, args.max_len)
    test_loader = DataLoader(test_dataset, args.batch_size)

    model.eval()
    preds = []
    # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
    with torch.no_grad():
        for data in tqdm(test_loader, dynamic_ncols=True):
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long()
            }
            outputs = model(inputs).cpu().numpy()
            pred = np.argmax(outputs, axis=-1)
            preds.extend(pred)
    
    # 修正标签
    for idx in homoIdx:
        preds[idx] = 1

    pd.DataFrame({'label': [i for i in preds]})\
        .to_csv(args.model_save_path + args.model_timestamp + "/result.csv", index=False, header=None)
    time_end = time()
    print(f'finish {time_end - time_start}s')


# ================================================================== #
#                   根据置信度生成伪标签 需要模型最后层softmax输出          #
# ================================================================== #
def creadte_fake_label(model, prob=0.98):
    set_seed(args.seed)
    test_data = load_data(args.test_data_path)
    test_dataset = FineTuneDataset(test_data, args.max_len)
    test_loader = DataLoader(test_dataset, args.batch_size)
    preds = []
    with torch.no_grad():
        for data in tqdm(test_loader, position=0, dynamic_ncols=True, desc='输出伪标签'):
            inputs = {
                'input_ids': data['input_ids'].to(args.device).long(),
                'attention_mask': data['attention_mask'].to(args.device).long(),
                'token_type_ids': data['token_type_ids'].to(args.device).long()
            }
            outputs = model.forward(inputs)
            preds.extend(outputs.cpu().numpy())
    # 记录满足指定置信度的伪标签样本
    fake_index, fake_label = [False] * len(preds), []
    for index, value in enumerate(preds):
        flag = False
        for i in range(2):
            if value[i] >= prob:
                flag = True
        if flag:
            fake_index[index] = True
            fake_label.append(np.argmax(value))
    # 处理输出
    result = pd.read_csv(args.test_data_path, sep='\t', header=None)
    result = result[fake_index]
    result['label'] = fake_label
    result.to_csv(f'./data/fake_{prob}.csv', index=False, header=False)


def main():
    set_seed(args.seed)
    model = Model(args=args)
    # 训练模式
    if args.do_train:
        model = model.to(args.device)
        train(model)
    # 读取自预训练后训练
    elif args.do_train_after_pretrain:
        # 读取路径目录
        file_dir = args.pre_model_path + args.pre_model_timestamp
        file_list = os.listdir(file_dir)
        for name in file_list:
            if name == f'{args.model_type}.pth' or name.split('.')[-1] != 'pth':
                continue
            model_path = os.path.join(file_dir, name)
            if os.path.isfile(model_path) and name.split('-')[1] == f'epoch{args.pre_epoch}.pth':
                print('pretrain model: ', name)
                state_dict = torch.load(model_path, map_location='cuda')
                model.load_state_dict(state_dict, strict=False)
                model = model.to(args.device)
                train(model)
    # 预测
    elif args.do_predict:
        # 读取微调后的模型权重
        state_dict = torch.load(
            args.model_save_path + args.model_timestamp + f'/{args.model_type}_{args.struc}_best_model.pth',
            map_location='cuda')
        model.load_state_dict(state_dict)
        model = model.to(args.device)
        predict(model)

if __name__ == '__main__':
    main()
