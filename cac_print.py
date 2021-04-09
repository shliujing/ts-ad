import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matrixprofile as mp
import os
import csv

# 读取所有文件
data_path = './PhaseII/'
figure_path = './Figure/'
ONE_ROW_FILES = ['204_UCR_Anomaly_12412.txt', '206_UCR_Anomaly_25130.txt', '207_UCR_Anomaly_3165.txt',
                 '208_UCR_Anomaly_5130.txt', '225_UCR_Anomaly_81214.txt', '226_UCR_Anomaly_96123.txt',
                 '242_UCR_Anomaly_100000.txt', '243_UCR_Anomaly_100000.txt']


def read_csv_file():
    yseries = []
    index_dict = []
    index_normal = []
    files = os.listdir(data_path)
    # print(files) # 文件名
    for f in files:
        idx = f.split('/')[-1]
        # print(idx) # 每个长度
        csv_path = data_path + f
        df = pd.read_csv(csv_path, names=['values'])
        values = df['values'].values
        print(values.shape)
        yseries.append(values)
        index_dict.append(idx)
        index_normal.append(idx.split('_')[3].split('.')[0])
    return yseries, index_dict, index_normal


def show_figure(values, index_dict, index_normal):
    for index, value in enumerate(values):
        sp = value.shape[0]
        if (sp == 1):  # todo 行为1的数据，先跳过，暂时先不处理
            continue
        ids = np.arange(1, sp + 1, 1, int)
        max = np.max(value)  # max value index
        min = np.min(value)  # min value index
        plt.plot(ids, value, 'r--')
        plt.vlines(int(index_normal[index]), min, max, colors="b", linestyles="dashed")
        plt.title(index_dict[index])  # 添加图形标题
        plt.savefig(figure_path + index_dict[index] + '.jpg')
        plt.show()


def compute_an(values, window_size=100):
    res = []
    for index, value in enumerate(values):
        sp = value.shape[0]
        if (sp == 1):  # todo 行为1的数据，先跳过，暂时先不处理
            res.append([0, 1, 2])
            continue
        print('begin file ' + str(index+1) + ' with ' + str(sp) + ' points:')
        starttime = datetime.datetime.now()
        profile = mp.compute(value, window_size)  # todo
        re = mp.discover.discords(profile)['discords']  # todo
        res.append(re)
        endtime = datetime.datetime.now()
        spend_time = endtime - starttime
        print('end file ' + str(index+1) + ', spend time:' + str(spend_time))
    return res


def gen_result_file(res_avg):
    csvFile = open("./submissionsample.csv", 'w', newline='')
    try:
        writer = csv.writer(csvFile)
        writer.writerow(('No.', 'location'))
        for index, value in enumerate(res_avg):
            writer.writerow((index + 1, value))
    finally:
        csvFile.close()


print('shape:')
values, index_dict, index_normal = read_csv_file()

# print('\n data figure show')
# show_figure(values, index_dict, index_normal)

print('\n predict interval:')
res = compute_an(values)
print(res)  # 每段区间

print('\n sort interval:')
res_avg = []
for i in res:
    t = np.sort(i)
    res_avg.append(t[1])
print(res_avg)

print('\n generate file:')
gen_result_file(res_avg)
print('\n generate file done:')
