# -*- codeing =utf-8 -*-
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import dataGeniter
import network
import Utils
from FocalLoss import FocalLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import datetime
import os

SEED = 42  # 或其他固定值

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def loadData():
    # 读入数据
    data_HSI = sio.loadmat('D:/dataset/hsi+lidar/muufl_hsi.mat')['muufl_hsi']
    data_lidar = sio.loadmat('D:/dataset/hsi+lidar/muufl_lidar.mat')['muufl_lidar']
    labels = sio.loadmat('D:/dataset/hsi+lidar/muufl_gt.mat')['muufl_gt']
    return data_HSI, data_lidar, labels


epochs = 150
BATCH_SIZE = 64
Patch_size = 9
NUM_num = 10  # 每类的训练样本数量
LEARNING_RATE = 0.0005 # 学习率0.0001
RESULT_PATH = 'D:/project_result'
RESULT_NAME = 'muufl'


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except:
            pass

    # 处理NumPy数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 处理字典
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}

    # 处理列表和元组
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]

    # 处理基本数据类型
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    try:
        return float(obj) if isinstance(obj, (np.floating, float)) else int(obj)
    except:
        return str(obj)

def train(train_loader, test_loader, epochs, NUM_CLASSES, target_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = network.MMCTNet(num_classes=NUM_CLASSES).to(device)


    loss_func = FocalLoss(class_num=NUM_CLASSES)

    # 初始化最佳指标
    best_oa = 0.0
    best_aa = 0.0
    best_kappa = 0.0
    best_epoch = 0
    best_model_state = None

    # 初始化历史记录
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'oa': [],
        'aa': [],
        'kappa': [],
        'best_epoch': 0,
        'best_train_loss': float('inf'),
        'best_val_acc': 0.0,
        'best_oa': 0.0,
        'best_aa': 0.0,
        'best_kappa': 0.0
    }

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    try:
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for i, (data1, data2, target) in enumerate(train_loader):
                data1, data2, target = data1.to(device), data2.to(device), target.to(device)
                outputs, _ = net(data1, data2)
                loss = loss_func(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()

            avg_loss = epoch_loss / len(train_loader)
            train_acc = 100.0 * epoch_correct / epoch_total

            # 每10个epoch或最后一个epoch记录一次详细指标
            #if epoch % 10 == 0 or epoch == epochs - 1:
            if epoch == epochs - 1:
                # 在测试集上评估
                net.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                pre = np.array([], dtype=np.int32)
                y_test_list = []

                for i, (data1, data2, target) in enumerate(test_loader):
                    data1, data2, target = data1.to(device), data2.to(device), target.to(device)
                    with torch.no_grad():
                        outputs, _ = net(data1, data2)
                        loss_test = loss_func(outputs, target)
                        test_loss += loss_test.item()

                    _, predicted = torch.max(outputs.data, 1)
                    test_correct += (predicted == target).sum().item()
                    test_total += target.size(0)

                    # 修复：使用正确的方法将PyTorch张量转换为NumPy数组
                    predicted_np = predicted.cpu().numpy()
                    target_np = target.cpu().numpy()
                    pre = np.concatenate([pre, predicted_np], 0)
                    y_test_list.extend(target_np)

                # 计算测试指标
                test_acc = 100.0 * test_correct / test_total
                avg_test_loss = test_loss / len(test_loader)

                # 计算OA, AA, Kappa
                y_test = np.array(y_test_list)
                confusion = confusion_matrix(y_test, pre)
                oa = accuracy_score(y_test, pre) * 100
                each_acc, aa = AA_andEachClassAccuracy(confusion)
                aa = aa * 100
                kappa = cohen_kappa_score(y_test, pre) * 100

                # 记录历史
                history['epochs'].append(epoch)
                history['train_loss'].append(float(avg_loss))
                history['train_acc'].append(float(train_acc))
                history['val_loss'].append(float(avg_test_loss))
                history['val_acc'].append(float(test_acc))
                history['oa'].append(float(oa))
                history['aa'].append(float(aa))
                history['kappa'].append(float(kappa))

                print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Test Acc: {test_acc:.2f}%, OA: {oa:.2f}%, AA: {aa:.2f}%, Kappa: {kappa:.2f}%')

                # 更新最佳指标
                if test_acc > history['best_val_acc']:
                    history['best_epoch'] = epoch
                    history['best_train_loss'] = float(avg_loss)
                    history['best_val_acc'] = float(test_acc)
                    history['best_oa'] = float(oa)
                    history['best_aa'] = float(aa)
                    history['best_kappa'] = float(kappa)
                    best_model_state = net.state_dict().copy()
                    torch.save(best_model_state,
                               f'./best_model_run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
                    print(f'New best model saved at epoch {epoch}')
            else:
                # 对于非记录epoch，只打印基本信息
                print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')

        print('Finished Training')
        print(
            f'Best model at epoch {history["best_epoch"]}: OA = {history["best_oa"]:.4f}%, AA = {history["best_aa"]:.4f}%')

        # 加载最佳模型参数
        if best_model_state is not None:
            net.load_state_dict(best_model_state)

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        # 返回当前状态
        return net, device, history

    return net, device, history


def sampling1(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = int(max(ground_truth))
    for i in range(m + 1):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i
        ]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m + 1):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def select_traintest(groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}

    # 自动获取类别数量（排除0类，0通常是背景）
    unique_labels = np.unique(groundTruth)
    valid_labels = unique_labels[unique_labels > 0]  # 排除背景类
    NUM_CLASSES = len(valid_labels)

    # 自动生成target_names
    target_names = [str(int(label)) for label in valid_labels]

    # 自动生成amount列表
    amount = [NUM_num] * NUM_CLASSES

    print(f"自动检测到 {NUM_CLASSES} 个类别: {target_names}")

    for i, label in enumerate(valid_labels):
        indices = [
            j for j, x in enumerate(groundTruth.ravel().tolist()) if x == label
        ]
        np.random.shuffle(indices)
        labels_loc[i] = indices

        nb_val = min(len(indices), amount[i])
        # 确保不超出范围
        if len(indices) < nb_val:
            nb_val = len(indices)
            print(f"Warning: Class {label} has only {len(indices)} samples, using all as training")

        train[i] = indices[:nb_val]  # 取前 nb_val
        test[i] = indices[nb_val:]  # 其余作为测试集
        print(
            f"Class {label}: Total samples = {len(indices)}, Train samples = {len(train[i])}, Test samples = {len(test[i])}")

    train_indices = []
    test_indices = []
    for i in range(len(valid_labels)):
        train_indices += train[i]
        test_indices += test[i]

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # 打印总体统计信息
    total_samples = len(train_indices) + len(test_indices)
    print(f"\nOverall: Train samples = {len(train_indices)} ({len(train_indices) / total_samples * 100:.2f}%), "
          f"Test samples = {len(test_indices)} ({len(test_indices) / total_samples * 100:.2f}%)")

    return train_indices, test_indices, NUM_CLASSES, target_names


def create_data_loader():
    # 读入数据
    X1, X2, y = loadData()
    if X2.ndim == 3:
        print("LiDAR has extra channels, shape =", X2.shape)
        X2 = X2[:, :, 0]  # 保留第一通道

    patch_size = Patch_size
    PATCH_LENGTH = int((patch_size - 1) / 2)

    # reshape gt
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(int)

    TOTAL_SIZE = len(gt[gt > 0])
    ALL_SIZE = len(gt)

    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    print('Hyperspectral data shape: ', X1.shape)
    print('Lidar data shape: ', X2.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X1 = applyPCA(X1, numComponents=pca_components)
    print('Data shape after PCA: ', X1.shape)

    # 将数据变换维度： [m,n,k]->[m*n,k]
    X1_all_data = X1.reshape(np.prod(X1.shape[:2]), np.prod(X1.shape[2:]))
    X2_all_data = X2.reshape(np.prod(X2.shape[:2]), )
    gt = y.reshape(np.prod(y.shape[:2]), )
    gt = gt.astype(int)

    # 自动获取类别数量
    unique_labels = np.unique(gt)
    valid_labels = unique_labels[unique_labels > 0]
    NUM_CLASSES = len(valid_labels)
    print(f"Number of classes: {NUM_CLASSES}")

    # 数据标准化
    X1_all_data = preprocessing.scale(X1_all_data)
    data_X1 = X1_all_data.reshape(X1.shape[0], X1.shape[1], X1.shape[2])
    whole_data_X1 = data_X1
    padded_data_X1 = np.pad(whole_data_X1, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                            'constant', constant_values=0)

    X2_all_data = preprocessing.scale(X2_all_data)
    data_X2 = X2_all_data.reshape(X2.shape[0], X2.shape[1])
    whole_data_X2 = data_X2
    padded_data_X2 = np.pad(whole_data_X2, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH)),
                            'constant', constant_values=0)
    print('\n... ... create train & test data ... ...')

    # 接收返回的NUM_CLASSES和target_names
    train_indices, test_indices, NUM_CLASSES, target_names = select_traintest(gt)

    _, all_indices = sampling1(1, gt)
    _, total_indices = sampling1(1, gt)
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)

    print('\n-----Selecting Small Cube from the Original Cube Data-----')
    train_iter, test_iter, total_iter, all_iter = dataGeniter.generate_iter(
        TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
        ALL_SIZE, all_indices, whole_data_X1, whole_data_X2, PATCH_LENGTH, padded_data_X1, padded_data_X2,
        pca_components, BATCH_SIZE, gt)

    # 修返回NUM_CLASSES和target_names
    return train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices, NUM_CLASSES, target_names


def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    attn_maps = []  # 存储注意力图

    for (data1, data2, labels) in test_loader:
        data1, data2 = data1.to(device), data2.to(device)
        outputs, features = net(data1, data2)

        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test, attn_maps


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


# 接收target_names参数
def acc_reports(y_test, y_pred_test, target_names):
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names, output_dict=True)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    # 计算F1-score
    f1_scores = []
    for i in range(1, len(target_names) + 1):  # 根据target_names长度确定类别数
        f1_scores.append(classification[str(i)]['f1-score'])

    f1_macro = classification['macro avg']['f1-score']
    f1_weighted = classification['weighted avg']['f1-score']

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100, f1_scores, f1_macro * 100, f1_weighted * 100


if __name__ == '__main__':
    set_seed(SEED)
    NUM_REPEATS = 1

    oa_list = []
    kappa_list = []
    aa_list = []
    each_acc_list = []
    classification_list = []
    confusion_list = []
    f1_scores_list = []
    f1_macro_list = []
    f1_weighted_list = []

    # 存储最佳指标和历史记录
    best_oa_list = []
    best_aa_list = []
    best_kappa_list = []
    best_epoch_list = []
    history_list = []

    file_name = f"{RESULT_PATH}/{RESULT_NAME}_all_runs.txt"
    with open(file_name, 'w') as x_file:
        for repeat in range(NUM_REPEATS):
            print(f"======== Run {repeat + 1} / {NUM_REPEATS} ========")

            train_iter, test_iter, total_iter, all_iter, y, total_indices, all_indices, NUM_CLASSES, target_names = create_data_loader()

            tic1 = time.perf_counter()
            # 修改：接收返回的历史记录
            net, device, history = train(
                train_iter, test_iter, epochs=epochs,
                NUM_CLASSES=NUM_CLASSES,
                target_names=target_names
            )
            toc1 = time.perf_counter()

            # 保存历史记录到JSON文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            history_filename = f"{RESULT_PATH}/{RESULT_NAME}_run{repeat + 1}_{timestamp}.json"

            # 添加实验配置信息
            history['experiment_config'] = {
                'dataset': 'augsburg',
                'train_samples_per_class': NUM_num,
                'num_classes': NUM_CLASSES,
                'patch_size': Patch_size,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'total_epochs': epochs,
                'seed': SEED,
                'target_names': target_names
            }

            # 转换并保存历史记录
            try:
                serializable_history = convert_to_serializable(history)
                with open(history_filename, 'w') as f:
                    json.dump(serializable_history, f, indent=2)
                print(f'Training history saved to {history_filename}')
            except Exception as e:
                print(f"Error saving history: {e}")
                # 备用保存方法
                try:
                    with open(history_filename, 'w') as f:
                        json.dump(history, f, indent=2, default=str)
                    print(f'Training history saved with string conversion: {history_filename}')
                except Exception as e2:
                    print(f"Failed to save history even with string conversion: {e2}")


            # 存储最佳指标
            best_oa = history['best_oa']
            best_aa = history['best_aa']
            best_kappa = history['best_kappa']
            best_epoch = history['best_epoch']

            best_oa_list.append(best_oa)
            best_aa_list.append(best_aa)
            best_kappa_list.append(best_kappa)
            best_epoch_list.append(best_epoch)
            history_list.append(history)

            # 使用最佳模型进行最终测试
            tic2 = time.perf_counter()
            y_pred_test, y_test, attn_maps = test(device, net, test_iter)
            toc2 = time.perf_counter()

            # 传递target_names参数
            classification, oa, confusion, each_acc, aa, kappa, f1_scores, f1_macro, f1_weighted = acc_reports(y_test,
                                                                                                               y_pred_test,
                                                                                                               target_names)

            Training_Time = toc1 - tic1
            Test_time = toc2 - tic2

            # 累积到列表
            oa_list.append(oa)
            kappa_list.append(kappa)
            aa_list.append(aa)
            each_acc_list.append(each_acc)
            classification_list.append(classification)
            confusion_list.append(confusion)
            f1_scores_list.append(f1_scores)
            f1_macro_list.append(f1_macro)
            f1_weighted_list.append(f1_weighted)

            # 更新历史记录文件添加最终测试结果
            history['final_results'] = {
                'final_accuracy': float(oa),
                'final_aa': float(aa),
                'final_kappa': float(kappa),
                'final_f1_macro': float(f1_macro),
                'final_f1_weighted': float(f1_weighted),
                'train_time': float(Training_Time),
                'test_time': float(Test_time),
                'each_class_accuracy': [float(acc) for acc in each_acc],
                'each_class_f1_score': [float(f1) for f1 in f1_scores]
            }

            # 重新保存更新后的历史记录
            try:
                serializable_history = convert_to_serializable(history)
                with open(history_filename, 'w') as f:
                    json.dump(serializable_history, f, indent=2)
                print(f'Updated training history with final results saved to {history_filename}')
            except Exception as e:
                print(f"Error updating history with final results: {e}")

            # 写入本次结果
            x_file.write(f"====== Run {repeat + 1} ======\n")
            x_file.write('Best Epoch: {}\n'.format(best_epoch))
            x_file.write('Best OA (%): {:.4f}\n'.format(best_oa))
            x_file.write('Best AA (%): {:.4f}\n'.format(best_aa))
            x_file.write('Best Kappa (%): {:.4f}\n'.format(best_kappa))
            x_file.write('Training Time (s): {}\n'.format(Training_Time))
            x_file.write('Test Time (s): {}\n'.format(Test_time))
            x_file.write('Final Test OA (%): {:.4f}\n'.format(oa))
            x_file.write('Final Test AA (%): {:.4f}\n'.format(aa))
            x_file.write('Final Test Kappa (%): {:.4f}\n'.format(kappa))
            x_file.write('Each class accuracy (%): {}\n'.format(each_acc))
            x_file.write('Each class F1-score (%): {}\n'.format(f1_scores))
            x_file.write('Macro F1-score (%): {:.4f}\n'.format(f1_macro))
            x_file.write('Weighted F1-score (%): {:.4f}\n'.format(f1_weighted))
            x_file.write('Classification report:\n{}\n'.format(
                classification_report(y_test, y_pred_test, digits=4, target_names=target_names)))
            x_file.write('Confusion matrix:\n{}\n\n'.format(confusion))

        x_file.write("\n======= Best Metrics Summary over {} runs =======\n".format(NUM_REPEATS))
        x_file.write("Best OA mean: {:.4f} ± {:.4f}\n".format(np.mean(best_oa_list), np.std(best_oa_list)))
        x_file.write("Best AA mean: {:.4f} ± {:.4f}\n".format(np.mean(best_aa_list), np.std(best_aa_list)))
        x_file.write("Best Kappa mean: {:.4f} ± {:.4f}\n".format(np.mean(best_kappa_list), np.std(best_kappa_list)))
        x_file.write("Best epochs: {}\n".format(best_epoch_list))

