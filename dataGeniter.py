# -*- codeing =utf-8 -*-
# -*- codeing =utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data
from tqdm import tqdm

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def select_small_cubic_1(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    # 使用 float32，降低内存占用
    small_cubic_data = np.zeros(
        (data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension),
        dtype=np.float32
    )
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in tqdm(range(len(data_assign)), desc="Extracting HSI patches (train)"):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data


def select_small_cubic_2(data_size, data_indices, whole_data, patch_length, padded_data):
    # 使用 float32，降低内存占用
    small_cubic_data = np.zeros(
        (data_size, 2 * patch_length + 1, 2 * patch_length + 1),
        dtype=np.float32
    )
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in tqdm(range(len(data_assign)), desc="Extracting LiDAR/SAR patches (train)"):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data


class StreamingPatchDataset(Data.Dataset):
    """
    按需提取 patch 的数据集，用于 test / total / all。
    - 返回：
        HSI:   [1, C, ph, pw]
        LiDAR: [1, ph, pw]
        label: (gt - 1)
    """
    def __init__(self, indices, rows, cols,
                 padded_hsi, padded_lidar,
                 patch_len, bands, gt):
        super().__init__()
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rows = rows
        self.cols = cols
        # 直接保证 float32
        self.padded_hsi = padded_hsi.astype(np.float32, copy=False)      # [H+2P, W+2P, C]
        self.padded_lidar = padded_lidar.astype(np.float32, copy=False)  # [H+2P, W+2P]
        self.patch_len = patch_len
        self.bands = bands
        self.gt = gt.astype(np.int64)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        flat = int(self.indices[idx])
        r = flat // self.cols
        c = flat % self.cols
        P = self.patch_len
        pr = r + P
        pc = c + P

        # HSI patch: (ph, pw, C) -> (C, ph, pw) -> [1, C, ph, pw]
        patch_hsi = self.padded_hsi[pr-P:pr+P+1, pc-P:pc+P+1, :]        # (ph, pw, C)
        patch_hsi = np.transpose(patch_hsi, (2, 0, 1))                  # (C, ph, pw)
        patch_hsi = np.expand_dims(patch_hsi, 0).astype(np.float32)     # [1, C, ph, pw]

        # LiDAR/SAR patch: (ph, pw) -> [1, ph, pw]
        patch_lidar = self.padded_lidar[pr-P:pr+P+1, pc-P:pc+P+1]
        patch_lidar = np.expand_dims(patch_lidar, 0).astype(np.float32) # [1, ph, pw]

        y = int(self.gt[flat]) - 1
        return (torch.from_numpy(patch_hsi),
                torch.from_numpy(patch_lidar),
                torch.tensor(y, dtype=torch.long))


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, ALL_SIZE, all_indices,
                  whole_data1, whole_data2, PATCH_LENGTH, padded_data1, padded_data2, INPUT_DIMENSION, batch_size, gt):

    # ===== 训练集：沿用原有“预展开”方式（数量小，速度快） =====
    X1_train_data = select_small_cubic_1(
        TRAIN_SIZE, train_indices, whole_data1, PATCH_LENGTH, padded_data1, INPUT_DIMENSION
    )
    print(X1_train_data.shape)
    X2_train_data = select_small_cubic_2(
        TRAIN_SIZE, train_indices, whole_data2, PATCH_LENGTH, padded_data2
    )
    print(X2_train_data.shape)

    X1_train = X1_train_data.transpose(0, 3, 1, 2)  # -> [N, C, H, W]
    X1_train_tensor = torch.from_numpy(X1_train).float().unsqueeze(1)       # [N,1,C,H,W]
    X2_train_tensor = torch.from_numpy(X2_train_data).float().unsqueeze(1)  # [N,1,H,W]
    y_train = gt[train_indices] - 1
    y_train_tensor = torch.from_numpy(y_train).long()

    torch_dataset_train = Data.TensorDataset(X1_train_tensor, X2_train_tensor, y_train_tensor)

    # ===== 流式数据集：test / total / all —— 不再预展开，避免爆内存 =====
    H, W = whole_data1.shape[0], whole_data1.shape[1]
    bands = INPUT_DIMENSION

    torch_dataset_test = StreamingPatchDataset(
        test_indices, H, W, padded_data1, padded_data2, PATCH_LENGTH, bands, gt
    )
    torch_dataset_total = StreamingPatchDataset(
        total_indices, H, W, padded_data1, padded_data2, PATCH_LENGTH, bands, gt
    )
    torch_dataset_all = StreamingPatchDataset(
        all_indices, H, W, padded_data1, padded_data2, PATCH_LENGTH, bands, gt
    )

    # ===== DataLoader =====
    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    total_iter = Data.DataLoader(
        dataset=torch_dataset_total,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_iter, test_iter, total_iter, all_iter
