# coding:utf-8
"""
@file: .py
@author: dannyXSC
@ide: PyCharm
@createTime: 2021年12月20日 20点45分
@Function: 请描述这个py文件的作用
"""

from transformer.process.bgl_preprocessor import *
from transformer.encoder.Encoder import Encoder
from transformer.learning.self_atten_train import SelfAttentive, make_src_mask
import torch

MODEL_OUTPUT_PATH = "./transformer/output/model.pt"
ORIGIN_LOG_PATH = "./transformer/output/BGL_2k.log"
INDEX_VEC_OUT_PATH = "./transformer/output/index_vec"
PATTERN_VEC_OUT_PATH = "./transformer/output/pattern_vec"

N_HEADS = 4
INPUT_DIM = 300  # 就是INPUT_DIM
HID_DIM = 512
OUTPUT_DIM = 300
N_ENCODERS = 3
FEEDFORWARD_DIM = 2048
DROPOUT_RATE = 0.5
PAD_IDX = 1
EMM_IDX = 0
BATCH_SIZE = 10

INDEX_TO_TENSOR = []
pattern_to_indexList = {}


# def pattern_to_vec():
#     """
#     生成index_to_tensor和pattern_to_index两个列表，前者是"索引-词向量"列表
#     （注：未纳入[EMBEDDING]和PAD，即词向量从0开始索引)
#     后者是"日志序列id-词语表索引"列表，
#     :return: index_to_tensor和pattern_to_index两个列表
#     """
#     global INDEX_TO_TENSOR, pattern_to_indexList
#     INDEX_TO_TENSOR, pattern_to_indexList, _ = pattern_to_vec_bgl(
#         ORIGIN_LOG_PATH, WORD_VEC_PATH, pattern_vec_out_path,
#         INDEX_VEC_OUT_PATH, PAD_IDX, EMM_IDX)


def evaluateEpsilon(model, index_vec_path, iterator, epsilon):
    model.eval()

    pred_res = []

    TP = 0
    TN = 0
    FN = 0
    FP = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = torch.tensor(batch[0], dtype=torch.float, requires_grad=True)
            trg = torch.tensor(batch[1],
                               dtype=torch.float,
                               requires_grad=False)

            # output = model(src, INDEX_TO_TENSOR)
            src_mask = make_src_mask(src, PAD_IDX)
            src = log_index_sequence_to_vec(src, index_vec_path)
            output = model(src, src_mask)
            # print("output", output)

            # output = [batch size,1]
            zero = torch.zeros(output.shape)
            one = torch.ones(output.shape)

            pred_choice = torch.where(output >= epsilon, one, zero)

            for pred in pred_choice:
                pred_res.append(pred)

            TP += torch.sum((pred_choice == 1) & (trg == 1))

            TN += torch.sum((pred_choice == 0) & (trg == 0))

            FN += torch.sum((pred_choice == 0) & (trg == 1))

            FP += torch.sum((pred_choice == 1) & (trg == 0))

            print(f"第{i}组测试")

    return pred_res, TP, TN, FN, FP


def get_iterator(id_list, label_list):
    iterator = []
    with open(PATTERN_VEC_OUT_PATH) as f:
        pattern = json.load(f)
        cnt = 0
        batch_info = [[], []]
        for i in range(len(id_list)):
            batch_info[0].append(pattern[id_list[i]])
            batch_info[1].append(label_list[i])
            cnt += 1
            if cnt >= BATCH_SIZE:
                iterator.append(batch_info)
                cnt = 0
                batch_info = [[], []]
                continue
    return iterator


def process_log_list(log_list):
    id_list = []
    label_list = []
    for log in log_list:
        word_vec = log.split(" ")
        id_list.append(word_vec[1])
        # 正常的是0 不正常的是1
        label_list.append(0 if word_vec[0] == '-' else 1)

    return id_list, label_list


def transformer_test(log_list):
    id_list, label_list = process_log_list(log_list)
    TEST_ITERATOR = get_iterator(id_list, label_list)

    # 获得模型
    enc = Encoder(
        n_heads=N_HEADS,
        input_dim=INPUT_DIM,
        hid_dim=HID_DIM,  # 修改
        output_dim=OUTPUT_DIM,
        n_encoders=N_ENCODERS,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout_rate=DROPOUT_RATE,
        device='cpu')

    # 获得模型
    model = SelfAttentive(enc, PAD_IDX)

    # 获得模型参数
    model.load_state_dict(torch.load(MODEL_OUTPUT_PATH))

    pred_res, TP, TN, FN, FP = evaluateEpsilon(model=model,
                                               index_vec_path=INDEX_VEC_OUT_PATH,
                                               iterator=TEST_ITERATOR,
                                               epsilon=10000)

    print( TP, TN, FN, FP)

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    print(pred_res)
    print(f'| Test Precision: {p:.3f} |')
    print(f'| Test Recall: {r:.3f} |')
    print(f'| Test F-Score: {F1:.3f} |')
    print(f'| Test Accuracy: {acc:.3f} |')

    return pred_res
