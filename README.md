## 总体流程
预训练(先不做):
sh pretrain.sh

微调：
sh finetune.sh

预测:
sh predict.sh

## 代码目录及功能介绍
1.common_utils --常用工具包
 -- convert_nezha_original_tf_checkpoint_to_pytorch.py  tf模型转torch模型
 -- MyDataset.py    定义预训练以及微调环节的dataset类
 -- optimizer.py    Lookahead优化器
 -- util.py         加载数据、获取文件存储地址、日志、设置随机种子、截断句子、FGM对抗训练、Focalloss损失函数、Diceloss损失函数等功能性工具包

2.data        --数据文件

3.models      -- 模型定义
 -- downstream_model.py  bert下接模型结构，如idcnn
 -- finetune_model.py    微调用的Model类

4.pretrain_model_utils  --预训练常用工具包
 -- nezha  就是nezha官方提供的包

5.finetune.py  -- 微调   实现train、eval、predict

6.finetune_args.py --微调代码的超参数设置

7.preprocess.py  --就是统计一下数据的句子长，在最开始的时候纵览一下，方便后续对句长、batchsize参数的调整

8.pretrain.py --实现对语料的预训练

9.pretrain_args.py -- 预训练代码的超参数设置

10.vote.py --最后的投票处理，可以先不着急






