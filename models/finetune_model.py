import torch.nn.functional as F
from models.downstream_model import IDCNN
import torch.nn as nn
import torch
from transformers import BertModel, BertConfig
from pretrain_model_utils.nezha.configuration_nezha import NeZhaConfig
from pretrain_model_utils.nezha.modeling_nezha import NeZhaModel
from finetune_args import pretrain_model_path

# ================================================================== #
#                        定义模型                                     #
# ================================================================== #
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # 定义预训练模型
        if args.model_type.split('_')[0] == 'nezha':
            nezha_config = NeZhaConfig.from_json_file(pretrain_model_path[args.model_type] + "config.json")
            nezha_config.output_hidden_states = True
            nezha_config.max_position_embeddings = 1024
            self.bert = NeZhaModel.from_pretrained(pretrain_model_path[args.model_type], config=nezha_config)
        else:
            bert_config = BertConfig.from_json_file(pretrain_model_path[args.model_type] + "config.json")
            bert_config.output_hidden_states = True
            self.bert = BertModel.from_pretrained(pretrain_model_path[args.model_type], config=bert_config)

        for param in self.bert.parameters():
            param.requires_grad = True
            
        if not args.use_avg:
            args.avg_size = 1
        self.args = args
        #   下游结构
        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'idcnn':
            self.idcnn = IDCNN(input_size=768, filters=64)
            self.fc = nn.Linear(32 + 1 - args.avg_size, args.num_classes)
            
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])
        self.softmax = nn.Softmax(dim=1)
        if args.use_dynamic_fusion:
            self.classifier = nn.Linear(args.bert_hidden_size, 1)
        
    def forward(self, x):
        bert_output = self.bert(**x)
        # bert的输出
        sequence_out = bert_output.last_hidden_state # (batch_size, max_len, 768)
        pooled_output = bert_output.pooler_output # (batch_size, 768)
        encoded_layers = bert_output.hidden_states # 13个编码器输出的状态向量 13 x (batch_size, max_len, 768)
        # 是否动态融合bert特征
        if self.args.use_dynamic_fusion:
            output = self.get_dym_layer(encoded_layers)
        else:
            output = sequence_out
        # 下游结构
        if self.args.struc == 'cls':
            output = output[:, 0, :]  # cls
        else:
            if self.args.struc == 'bilstm':
                _, hidden = self.bilstm(output)
                last_hidden = hidden[0].permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.lstm_dim * 2)
            elif self.args.struc == 'bigru':
                _, hidden = self.bigru(output)
                last_hidden = hidden.permute(1, 0, 2)
                output = last_hidden.contiguous().view(-1, self.args.gru_dim * 2)
            elif self.args.struc == 'idcnn':
                output = self.idcnn(output)
                output = torch.mean(output, dim=1)
        # 是否平均池化
        if self.args.use_avg:
            if self.args.struc == 'idcnn':
                output = F.avg_pool1d(output.unsqueeze(1), kernel_size=32, stride=1).squeeze(1)
            else:
                output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)
        # dropout
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        output = torch.sigmoid(output)
        # output = self.softmax(output)

        return output

    def get_dym_layer(self, all_encoder_layers):
        layer_logits = []
        all_encoder_layers = all_encoder_layers[1:] # 取后12个编码器输出
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer)) # (batch_size, max_len, 1)
        layer_logits = torch.cat(layer_logits, 2) # 按第三个维度拼接 (batch_size, max_len, 12)
        layer_dist = torch.softmax(layer_logits, dim=-1) # 对12维度做Softmax, (batch_size, max_len, 12)

        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)

        return pooled_output


class ModelForDynamicLen(nn.Module):
    def __init__(self, bert_config, args):
        super(ModelForDynamicLen, self).__init__()
        MODEL_NAME = {'nezha_wwm': 'NeZhaModel', 'nezha_base': 'NeZhaModel', 'roberta': 'BertModel'}
        self.bert = globals()[MODEL_NAME[args.model_type]](config=bert_config)
        self.args = args

        if args.struc == 'cls':
            self.fc = nn.Linear(768 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bilstm':
            self.bilstm = nn.LSTM(768, args.lstm_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.lstm_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'bigru':
            self.bigru = nn.GRU(768, args.gru_dim, bidirectional=True, num_layers=1, batch_first=True)
            self.fc = nn.Linear(args.gru_dim * 2 + 1 - args.avg_size, args.num_classes)
        elif args.struc == 'idcnn':
            self.idcnn = IDCNN(input_size=768, filters=64)
            self.fc = nn.Linear(32 + 1 - args.avg_size, args.num_classes)
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(args.dropout_num)])

    def forward(self, input_ids):
        output = None
        if self.args.struc == 'cls':
            output = torch.stack(
                [self.bert(input_id.to(self.args.device))[0][0][0]
                 for input_id in input_ids])

        if self.args.AveragePooling:
            output = F.avg_pool1d(output.unsqueeze(1), kernel_size=self.args.avg_size, stride=1).squeeze(1)

        # output = self.dropout(output)
        if self.args.dropout_num == 1:
            output = self.dropouts[0](output)
            output = self.fc(output)
        else:
            out = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(output)
                    out = self.fc(out)
                else:
                    temp_out = dropout(output)
                    out = out + self.fc(temp_out)
            output = out / len(self.dropouts)

        return output
