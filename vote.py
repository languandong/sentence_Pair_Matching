import pandas as pd
from collections import Counter


def main():
    total_fold5_tag_list = []
    for i in range(1, 6):
        file_path = f"./data/KFold_dataset/fold{i}/result{i}.csv"
        df = pd.read_csv(file_path, header=None)
        each_fold_tag_list = df[0].tolist()
        total_fold5_tag_list.append(each_fold_tag_list)
    new_tag_list = []
    for index in range(len(total_fold5_tag_list[0])):
        new_tag_list.append([total_fold5_tag_list[0][index]] + \
                            [total_fold5_tag_list[1][index]] + \
                            [total_fold5_tag_list[2][index]] + \
                            [total_fold5_tag_list[3][index]] + \
                            [total_fold5_tag_list[4][index]])
    vote_tag_list, id_list = [], []
    for idx, tag in enumerate(new_tag_list):
        id_list.append(idx)
        count = Counter(tag)
        count = sorted(count.items(), key=lambda x: -x[1])
        vote_tag_list.append(count[0][0])

    pd.DataFrame({'label': vote_tag_list}).to_csv('vote_result.csv', index=False, header=False)


if __name__ == '__main__':
    main()
