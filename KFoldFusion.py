import pandas as pd
from finetune_args import args, model_map
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
from common_utils.optimizer import build_optimizer
import numpy as np
import torch
import torch.nn as nn
import os
from common_utils.util import load_data, get_save_path, get_logger, set_seed, get_homophonesIdx, EMA
from common_utils.MyDataset import FineTuneDataset, BlockShuffleDataset
from common_utils.DataLoaders import BlockShuffleDataLoader
from models.finetune_model import Model
import random

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

data_path = './data/KFold_dataset'

time_start = time()
for fold in range(1,6):
  print('第{}折开始'.format(fold))
  # 创建模型保存路径以及记录配置的参数
  get_save_path(args)
  # 创建日志对象，写入训练效果
  logger = get_logger(args.model_save_path + '/finetune.log')
  # 加载数据file:/F:/deepLearning/ner4torch/args.py
  train_data = load_data(data_path + f'/fold{fold}/train')
  valid_data = load_data(data_path + f'/fold{fold}/val')
  train_dataset = BlockShuffleDataset(train_data, args.maxlen)
  valid_dataset = BlockShuffleDataset(valid_data, args.maxlen)
  train_loader = BlockShuffleDataLoader(train_dataset, batch_size=args.batch_size, is_shuffle=True,
                                        sort_key=lambda x: len(x[0]) + len(x[1]),
                                        collate_fn=train_dataset.collate)
  valid_loader = BlockShuffleDataLoader(valid_dataset, batch_size=args.batch_size, is_shuffle=False,
                                        sort_key=lambda x: len(x[0]) + len(x[1]),
                                        collate_fn=valid_dataset.collate)
  print(f'第{fold}折数据加载完成')
  # ================模型
  set_seed(args.seed)
  model = Model(args=args)
  # 读取已有预训练模型的参数
  state_dict = torch.load(model_map[args.model_type]['model_path'], map_location='cuda')
  # 加载预训练参数
  model.load_state_dict(state_dict, strict=False)
  model = model.to(args.device)

  # 优化器
  optimizer, scheduler = build_optimizer(args, model,
                                         total_steps=len(train_loader) * (args.epoch + 1))
  # 损失函数
  criterion = nn.BCELoss()
  # EMA
  ema = EMA(model, 0.999)
  ema.register()
  # ==================训练
  print(f'第{fold}折训练开始')
  best_acc = 0.
  early_stop = 0
  model.zero_grad()
  for epoch in range(args.epoch):
    model.train()
    losses, acc_list = [], []
    pbar = tqdm(train_loader, ncols=150, desc='训练中')
    # 一个batch的训练
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
      # 输出和计算损失
      outputs = model.forward(inputs)
      loss = criterion(outputs, data['label'])
      # 反向传播
      loss.backward()
      # 更新参数
      optimizer.step()

      if args.warmup:
        scheduler.step()

      ema.update()

      losses.append(loss.cpu().detach().numpy())
      output_array = outputs.cpu().detach().numpy()
      label_array = data['label'].cpu().detach().numpy()
      acc_list.extend(np.argmax(output_array, axis=1) == np.argmax(label_array, axis=1))
      pbar.set_description(
        f'fold:{fold} epoch:{epoch + 1}/{args.epoch} lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.7f} loss:{np.mean(losses):.4f} acc:{(np.sum(acc_list) / len(acc_list)):.3f}')

    ema.apply_shadow()

    # ======================验证
    print(f'第{fold}折验证开始')
    model.eval()

    true, preds = [], []
    pbar = tqdm(valid_loader, ncols=150)
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

  #====================预测
  print(f'第{fold}折测试开始')
  test_data = load_data(args.test_data_path)
  homoIdx = get_homophonesIdx(args.test_data_path)

  test_dataset = FineTuneDataset(test_data, args.maxlen)
  test_loader = DataLoader(test_dataset, args.batch_size)

  model.eval()
  preds = []
  # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
  with torch.no_grad():
    for data in tqdm(test_loader, ncols=150):
      inputs = {
        'input_ids': data['input_ids'].to(args.device).long(),
        'attention_mask': data['attention_mask'].to(args.device).long(),
        'token_type_ids': data['token_type_ids'].to(args.device).long()
      }
      outputs = model(inputs)
      pred = np.argmax(outputs.cpu().numpy(), axis=-1)
      preds.extend(pred)

  # 修正标签
  for idx in homoIdx:
    preds[idx] = 1

  pd.DataFrame({'label': [i for i in preds]}).to_csv(data_path + f'/fold{fold}/result{fold}.csv', index=False, header=None)

time_end = time()
print(f'finish {(time_end - time_start)/3600}h')
