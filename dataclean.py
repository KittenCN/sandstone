import numpy as np
from tqdm import tqdm

train_num = 0.8
ori_x = np.array(np.load('X_train_ori.npy')).astype(np.float32)
ori_y = np.array(np.load('y_train_ori.npy')).astype(np.float32)
ori_x_array = np.array(ori_x.reshape(-1, 1, 150, 150))
ori_x_train = ori_x_array[:int(ori_x_array.shape[0] * train_num)]
ori_y_train = ori_y[:int(ori_y.shape[0] * train_num)]
ori_x_test = ori_x_array[int(ori_x_array.shape[0] * train_num):]
ori_y_test = ori_y[int(ori_y.shape[0] * train_num):]

pbar = tqdm(total=ori_x_train.shape[0] * 3)
_ori_x_train = ori_x_train.copy()
for i in range(1, 4):
    for index, item in enumerate(_ori_x_train):
        pbar.update(1)
        ori_x_train = np.append(ori_x_train, np.rot90(item, i).reshape(-1,1,150,150), axis=0)
        ori_y_train = np.append(ori_y_train, ori_y_train[index])
pbar.close()

y_data = []
_y_data = []
for item in ori_y_train:
    binary = [0.0, 0.0, 0.0, 0.0]
    binary[int(item)] = 1.0
    # _y_data.append(binary)
    # y_data.append(_y_data)
    y_data.append(binary)
ori_y_train = np.array(y_data).astype(np.float32)
y_data = []
_y_data = []
for item in ori_y_test:
    binary = [0.0, 0.0, 0.0, 0.0]
    binary[int(item)] = 1.0
    # _y_data.append(binary)
    # y_data.append(_y_data)
    y_data.append(binary)
ori_y_test = np.array(y_data).astype(np.float32)

np.save('X_train_split.npy', ori_x_train)
np.save('y_train_split.npy', ori_y_train)
np.save('x_test_split.npy', ori_x_test)
np.save('y_test_split.npy', ori_y_test)
print(ori_x_train.shape)
print(ori_y_train.shape)
print(ori_x_test.shape)
print(ori_y_test.shape)