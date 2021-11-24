import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertModel
from pretrain_model_utils.nezha.configuration_nezha import NeZhaConfig
from pretrain_model_utils.nezha.modeling_nezha import NeZhaModel
from finetune_args import pretrain_model_path
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertOnlyMLMHead



# 定义model
class Model(nn.Module):
    def __init__(self, args, resume=False):
        super(Model, self).__init__()
        # 定义预训练模型
        if args.model_type.split('_')[0] == 'nezha':
            self.config = NeZhaConfig.from_json_file(pretrain_model_path[args.model_type] + "config.json")
            self.config.output_hidden_states = True
            self.config.max_position_embeddings = 1024
            self.bert = NeZhaModel.from_pretrained(pretrain_model_path[args.model_type], config=self.config)
        else:
            self.config = BertConfig.from_json_file(pretrain_model_path[args.model_type] + "config.json")
            self.config.output_hidden_states = True
            self.bert = BertModel.from_pretrained(pretrain_model_path[args.model_type], config=self.config)

        self.cls = BertOnlyMLMHead(self.config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        masked_lm_labels = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            mask = (labels != -100)
            masked_lm_loss = loss_fct(prediction_scores[mask].view(-1, self.config.vocab_size), labels[mask].view(-1))
            outputs = (masked_lm_loss,) + outputs
        return outputs  # (ltr_lm_loss), (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

