from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import random
from torch.utils.data import Dataset, DataLoader
from itertools import chain


# ================================================================== #
#                       分块shuffle，继承DataLoader类                  #
# ================================================================== #
class BlockShuffleDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        """
        Args:
            dataset: Dataset类的实例，其中必须包含dataset变量，并且该变量为一个list
            sort_key: 排序对象，即使用dataset元素中哪一个变量的长度进行排序
            sort_bs_num: 排序范围，即在多少个batch_size大小内进行排序，默认为None，表示对整个序列排序
            is_shuffle: 是否对分块后的内容，进行随机打乱，默认为True
            **kwargs:
        """
        assert isinstance(dataset.lines, list), "lines为Dataset类的实例，其中必须包含lines变量，并且该变量为一个list"
        super().__init__(dataset, **kwargs)
        self.sort_bs_num = sort_bs_num
        self.sort_key = sort_key
        self.is_shuffle = is_shuffle

    def __iter__(self):
        self.dataset.lines = self.block_shuffle(self.dataset.lines, self.batch_size, self.sort_bs_num,
                                                self.sort_key, self.is_shuffle)
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        # 将数据按照batch_size大小进行切分
            # 取出超过batch_size部分的数据
        tail_data = [] if (len(data) % batch_size) == 0 else data[-(len(data) % batch_size):]
            # 取出batch_size部分数据
        data = data[:(len(data) - len(tail_data))]
        assert len(data) % batch_size == 0
        # 获取真实排序范围
        sort_bs_num = len(data) // batch_size if sort_bs_num is None else sort_bs_num
        # 按照排序范围进行数据划分
        data = [data[i:i + sort_bs_num * batch_size] for i in range(0, len(data), sort_bs_num * batch_size)]
        # 在排序范围，根据排序函数进行降序排列
        data = [sorted(i, key=sort_key, reverse=True) for i in data]
        # 将数据根据batch_size获取batch_data
        data = list(chain(*data))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        # 判断是否需要对batch_data序列进行打乱
        if is_shuffle:
            random.shuffle(data)
        # 将tail_data填补回去
        data = list(chain(*data)) + tail_data
        return data
