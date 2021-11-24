from torch.utils.data import Dataset
import random
import numpy as np
import torch
from transformers import BertTokenizer
from common_utils.util import truncate_sequences
from finetune_args import args, pretrain_model_path

# ================================================================== #
#                       预训练数据迭代类                                #
# ================================================================== #
class PretrainDataset(Dataset):

    def __init__(self, corpus, seq_len: int, args):
        self.seq_len = seq_len - 3
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = BertTokenizer(
            vocab_file=pretrain_model_path[args.model_type] + 'vocab.txt',
            do_lower_case=True)
        self.vocab = {index: token for token, index in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2 = self.get_sentence(idx)
        text_pair = ['[CLS]'] + text1 + ['[SEP]'] + text2 + ['[SEP]']
        text_pair, output_ids = self.create_masked_lm_predictions(text_pair)
        token_ids = []
        for token in text_pair:
            temp_char = self.tokenizer.tokenize(token.lower())
            if temp_char:
                token_ids.append(self.tokenizer.convert_tokens_to_ids(temp_char[0]))
            else:
                token_ids.append(self.tokenizer.convert_tokens_to_ids("§"))
        segment_ids = [0] * len(token_ids)

        padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
        padding_label = [-100 for _ in range(self.seq_len + 3 - len(token_ids))]
        attention_mask = len(token_ids) * [1] + len(padding) * [0]
        token_ids.extend(padding), output_ids.extend(padding_label), segment_ids.extend(padding)
        attention_mask = np.array(attention_mask)
        token_ids = np.array(token_ids)
        segment_ids = np.array(segment_ids)
        output_ids = np.array(output_ids)
        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  'attention_mask': attention_mask,
                  "output_ids": output_ids}
        return output

    def get_sentence(self, idx):

        text1, text2, _ = self.lines[idx]
        text1, text2 = truncate_sequences(self.seq_len, -1, text1, text2)

        return text1, text2

    def create_masked_lm_predictions(self, text, masked_lm_prob=0.15, max_predictions_per_seq=21,
                                     rng=random.Random()):
        cand_indexes = []
        for (i, token) in enumerate(text):
            if token == '[CLS]' or token == '[SEP]':
                continue
            cand_indexes.append([i])

        output_tokens = text
        output_tokens_copy = output_tokens.copy()
        output_labels = [-100] * len(text)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(text) * masked_lm_prob))))

        # 不同 gram 的比例  **(改为3)**
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        # 每个 token 对应的三个 ngram
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        rng.shuffle(ngram_indexes)

        masked_lms = set()
        # 获取 masked tokens
        # cand_index_set 其实就是每个 token 的三个 ngram
        # 比如：[[[13]], [[13], [14]], [[13], [14], [15]]]
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # 根据 cand_index_set 不同长度 choice
            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            # [16, 17] = sum([[16], [17]], [])
            index_set = sum(cand_index_set[n - 1], [])
            # 处理选定的 ngram index ：80% MASK，10% 是原来的，10% 随机替换一个
            for index in index_set:
                masked_token = None
                if rng.random() < 0.8:
                    masked_token = '[MASK]'
                else:
                    if rng.random() < 0.5:
                        masked_token = text[index]
                    else:
                        masked_token = self.vocab[
                            rng.randint(0, self.tokenizer.vocab_size)]  # 取不到特殊字符
                        output_labels[index] = self.vocab.get(output_tokens[index], 1)
                temp_char = self.tokenizer.tokenize(output_tokens[index].lower())
                if temp_char:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids(temp_char[0])
                else:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids("§")
                output_tokens_copy[index] = masked_token
                masked_lms.add(index)

        return output_tokens_copy, output_labels


# ================================================================== #
#                        微调任务数据迭代类                             #
# ================================================================== #
class FineTuneDataset(Dataset):

    def __init__(self, corpus, seq_len: int):
        self.seq_len = seq_len - 3  # 序列长度 减去三个标志符
        self.lines = corpus  # 数据集     ( ([], [], []) , ....)
        self.corpus_lines = len(self.lines)  # 样本数量
        self.tokenizer = BertTokenizer(
            vocab_file=pretrain_model_path[args.model_type] + 'vocab.txt',
            do_lower_case=True)

    # 返回数据集的样本数
    def __len__(self):
        return self.corpus_lines

    # 返回指定idx样本的数据
    def __getitem__(self, idx):
        # 解析tuple [],[],[]
        text1, text2, label = self.lines[idx]
        # token_ids
        text1_token_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        text2_token_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        text1_token_ids, text2_token_ids = truncate_sequences(self.seq_len, -1, text1_token_ids, text2_token_ids)
        token_ids = [101] + text1_token_ids + [102] + text2_token_ids + [102]

        # 填补的序列
        padding = [0 for _ in range(self.seq_len + 3 - len(token_ids))]
        # token_type_ids 标识前后两句话
        segment_ids = [0] * (len(text1_token_ids) + 2) + [1] * (len(text2_token_ids) + 1)
        segment_ids.extend(padding)
        segment_ids = np.array(segment_ids)
        # attention_mask 有值标1
        attention_mask = np.array(len(token_ids) * [1] + len(padding) * [0])
        # input_ids
        token_ids.extend(padding)
        token_ids = np.array(token_ids)
        # label
        label = np.array(label)

        # 返回一个数据样本
        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  'attention_mask': attention_mask,
                  "label": label}

        return output


# ================================================================== #
#                     分块shuffle微调数据迭代类                         #
# ================================================================== #
class BlockShuffleDataset(Dataset):

    def __init__(self, corpus, seq_len: int):
        self.seq_len = seq_len - 3
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = BertTokenizer(
            vocab_file=pretrain_model_path[args.model_type] + 'vocab.txt',
            do_lower_case=True)
        self.vocab = {index: token for token, index in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2, label = self.get_sentence_and_label(idx)
        text1_token_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        text2_token_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        # 先删除句首字
        text1_token_ids, text2_token_ids = truncate_sequences(self.seq_len, -1, text1_token_ids, text2_token_ids)
        token_ids = [101] + text1_token_ids + [102] + text2_token_ids + [102]
        segment_ids = [0] * (len(text1_token_ids) + 2) + [1] * (len(text2_token_ids) + 1)

        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  "label": label}
        return output

    def get_sentence_and_label(self, idx):
        text1, text2, label = self.lines[idx]

        return text1, text2, label

    @classmethod
    # 调用不需要类实例化 不需要self参数 cls是表示自身类的参数
    # 定义padding的序列
    def padding_list(cls, ls, val, returnTensor=False):
        ls = ls[:]
        # 获得一个batch里最大序列长度
        max_len = max([len(i) for i in ls])
        # 填充batch每个样本到最大长度
        for i in range(len(ls)):
            ls[i] = ls[i] + [val] * (max_len - len(ls[i]))
        return torch.tensor(ls) if returnTensor else ls

    @classmethod
    # 聚合padding后的输入序列
    def collate(cls, batch):
        # 聚合一个batch的输入
        input_ids = [i['input_ids'] for i in batch]
        token_type_ids = [i['token_type_ids'] for i in batch]
        label = [i['label'] for i in batch]
        #
        input_ids = BlockShuffleDataset.padding_list(input_ids, 0, returnTensor=True)
        token_type_ids = BlockShuffleDataset.padding_list(token_type_ids, 0, returnTensor=True)
        label = torch.tensor(label)
        attention_mask = (input_ids != 0)
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'label': label}


# ================================================================== #
#                      分块shuffle的预训练数据类                        #
# ================================================================== #
class BlockShufflePretrainDataset(Dataset):

    def __init__(self, corpus, seq_len: int):
        self.seq_len = seq_len - 3
        self.lines = corpus
        self.corpus_lines = len(self.lines)
        self.tokenizer = BertTokenizer(
            vocab_file=pretrain_model_path[args.model_type] + 'vocab.txt',
            do_lower_case=True)
        self.vocab = {index: token for token, index in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        text1, text2 = self.get_sentence(idx)
        text_pair = ['[CLS]'] + text1 + ['[SEP]'] + text2 + ['[SEP]']
        text_pair, output_ids = self.create_masked_lm_predictions(text_pair)
        token_ids = []
        for token in text_pair:
            if token == '[CLS]' or '[SEP]':
                token_ids.append(self.tokenizer.convert_tokens_to_ids(token))
            else:
                temp_char = self.tokenizer.tokenize(token.lower())
                if temp_char:
                    token_ids.append(self.tokenizer.convert_tokens_to_ids(temp_char[0]))
                else:
                    token_ids.append(self.tokenizer.convert_tokens_to_ids("§"))
        segment_ids = [0] * (len(text1) + 2) + [1] * (len(text2) + 1)

        output = {"input_ids": token_ids,
                  "token_type_ids": segment_ids,
                  "output_ids": output_ids}
        return output

    def get_sentence(self, idx):

        text1, text2, _ = self.lines[idx]
        text1, text2 = list(text1), list(text2)
        text1, text2 = truncate_sequences(self.seq_len, -1, text1, text2)

        return text1, text2

    def create_masked_lm_predictions(self, text, masked_lm_prob=0.15, max_predictions_per_seq=512,
                                     rng=random.Random()):
        cand_indexes = []
        for (i, token) in enumerate(text):
            if token == '[CLS]' or token == '[SEP]':
                continue
            cand_indexes.append([i])

        output_tokens = text
        output_tokens_copy = output_tokens.copy()
        output_labels = [-100] * len(text)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(text) * masked_lm_prob))))

        # 不同 gram 的比例  **(改为3)**
        ngrams = np.arange(1, 3 + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, 3 + 1)
        pvals /= pvals.sum(keepdims=True)

        # 每个 token 对应的三个 ngram
        ngram_indexes = []
        for idx in range(len(cand_indexes)):
            ngram_index = []
            for n in ngrams:
                ngram_index.append(cand_indexes[idx:idx + n])
            ngram_indexes.append(ngram_index)
        rng.shuffle(ngram_indexes)

        masked_lms = set()
        # 获取 masked tokens
        # cand_index_set 其实就是每个 token 的三个 ngram
        # 比如：[[[13]], [[13], [14]], [[13], [14], [15]]]
        for cand_index_set in ngram_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # 根据 cand_index_set 不同长度 choice
            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] / pvals[:len(cand_index_set)].sum(keepdims=True))
            # [16, 17] = sum([[16], [17]], [])
            index_set = sum(cand_index_set[n - 1], [])
            # 处理选定的 ngram index ：80% MASK，10% 是原来的，10% 随机替换一个
            for index in index_set:
                masked_token = None
                if rng.random() < 0.8:
                    masked_token = '[MASK]'
                else:
                    if rng.random() < 0.5:
                        masked_token = text[index]
                    else:
                        try:
                          masked_token = self.vocab[
                              rng.randint(0, self.tokenizer.vocab_size)]  # 取不到特殊字符
                        except KeyError:
                          masked_token = self.vocab[4396] 
                        output_labels[index] = self.vocab.get(output_tokens[index], 1)
                temp_char = self.tokenizer.tokenize(output_tokens[index].lower())
                if temp_char:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids(temp_char[0])
                else:
                    output_labels[index] = self.tokenizer.convert_tokens_to_ids("§")
                output_tokens_copy[index] = masked_token
                masked_lms.add(index)

        return output_tokens_copy, output_labels

    @classmethod
    def padding_list(cls, ls, val, returnTensor=False):
        ls = ls[:]
        max_len = max([len(i) for i in ls])
        for i in range(len(ls)):
            ls[i] = ls[i] + [val] * (max_len - len(ls[i]))
        return torch.tensor(ls) if returnTensor else ls

    @classmethod
    def collate(cls, batch):
        input_ids = [i['input_ids'] for i in batch]
        token_type_ids = [i['token_type_ids'] for i in batch]
        output_ids = [i['output_ids'] for i in batch]
        input_ids = BlockShufflePretrainDataset.padding_list(input_ids, 0, returnTensor=True)
        token_type_ids = BlockShufflePretrainDataset.padding_list(token_type_ids, 0, returnTensor=True)
        output_ids = BlockShufflePretrainDataset.padding_list(output_ids, -100, returnTensor=True)
        attention_mask = (input_ids != 0)
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
                'output_ids': output_ids}


# ================================================================== #
#                      动态句长                                       #
# ================================================================== #
class FineTuneDatasetForDynamicLen(Dataset):

    def __init__(self, corpus, vocab: dict, args):
        self.vocab = vocab
        self.corpus = corpus
        self.args = args
        self.examples = self.processor()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.examples[0][idx], self.examples[1][idx]

    @staticmethod
    def collate_fn(data):
        data.sort(key=lambda x: - len(x[0]))
        data = list(zip(*data))
        id_batch = data[0]
        label_batch = np.array(data[1])
        return [id_batch, label_batch]

    def processor(self):
        id_list, label_list = [], []
        for text, label in self.corpus:
            text_ids = [self.vocab.get(t, 1) for t in text]
            token_ids = ([2] + text_ids + [3])
            id_list.append(torch.tensor([token_ids]).to(self.args.device).long())
            label_list.append(label)

        return [id_list, label_list]
