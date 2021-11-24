import pandas as pd
from finetune_args import args
from tqdm import tqdm
from transformers import BertTokenizer
from transformers.models.roberta.configuration_roberta import RobertaConfig
from pretrain_model_utils.nezha.configuration_nezha import NeZhaConfig
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import os
from common_utils.util import load_data, set_seed
from common_utils.MyDataset import FineTuneDataset
from models.finetune_model import Model
from torch.utils.data import DataLoader

set_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
MODEL_CONFIG = {'nezha_wwm': 'NeZhaConfig', 'nezha_base': 'NeZhaConfig', 'roberta': 'RobertaConfig'}

# 加载验证数据
valid_data = []
for valid_path in args.valid_data_path:
    valid_data += load_data(valid_path)
# 加载验证集原始数据
BQ_dev = pd.read_csv(args.valid_data_path[0], sep = '\t', header=None)
LCQMQ_dev = pd.read_csv(args.valid_data_path[1], sep = '\t', header=None)
OPPO_dev = pd.read_csv(args.valid_data_path[2], sep="\t", header=None)
df_tmp = BQ_dev.append(LCQMQ_dev)
all_dev = df_tmp.append(OPPO_dev)
# 验证集数据迭代类
valid_dataset = FineTuneDataset(valid_data, args.maxlen)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
# 读取预训练模型配置
model = Model(args=args)
# 读取已有模型的参数
state_dict = torch.load(
            args.model_save_path + args.model_timestamp + f'/{args.model_type}_{args.struc}_best_model.pth',
            map_location='cuda')
# 加载预训练参数
model.load_state_dict(state_dict, strict=False)
model = model.to(args.device)
model.eval()
true, preds = [], []
#tokens = []
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
    # pred和true中不同的布尔索引
    index = list(map(lambda x: x[0] != x[1], zip(preds, true)))

all_dev.insert(3,'pred',preds)
all_dev[index].to_csv('badcase.csv',index=False,header=False)



