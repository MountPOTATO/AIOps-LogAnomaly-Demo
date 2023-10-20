import json
import time
import pandas
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(nn.Module):
    def __init__(self, input_size_0, input_size_1, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm0 = nn.LSTM(input_size_0, hidden_size, num_of_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size_1, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, out_size)

    def forward(self, input_0, input_1):
        h0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        out_0, _ = self.lstm0(input_0, (h0_0, c0_0))
        h0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        out_1, _ = self.lstm1(input_1, (h0_1, c0_1))
        multi_out = torch.cat((out_0[:, -1, :], out_1[:, -1, :]), -1)
        fc_out = self.fc(multi_out)
        #print(fc_out)
        out = torch.sigmoid(fc_out)
        #print(out)
        return out

def loganomaly_run():
    hidden_size = 128
    num_of_layers = 2
    num_of_classes = 31
    num_epochs = 15

    window_length = 5
    input_size_sequential = 300
    input_size_quantitive = 31
    batch_size = 512
    test_batch_size = 512

    num_candidates = 6
    threshold = 3.714397962539806e-07

    logparser_structed_file = './loganomaly/drain_result/HDFS_split_40w.log_structured.csv'
    logparser_event_file = './loganomaly/drain_result/HDFS_split_40w.log_templates.csv'
    anomaly_label_file = './loganomaly/drain_result/anomaly_label.csv'

    sequential_directory = './loganomaly/sequential_files/'
    train_file_name = 'loganomaly_train_file'
    test_file_name = 'loganomaly_test_file'
    valid_file_name = 'loganomaly_valid_file'

    train_file = sequential_directory + train_file_name
    test_file = sequential_directory + test_file_name
    model_out_path = './loganomaly/output/'

    wordvec_file_path = 'G:\\crawl-300d-2M.vec'
    pattern_vec_out_path = './loganomaly/drain_result/pattern_vec'


    result_str, result_dict = do_predict(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
               model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt',
               test_file, pattern_vec_out_path, num_candidates, threshold)

    return result_str, result_dict


def load_model(input_size_1, input_size_2, hidden_size, num_layers, num_classes, model_path):
    model = Model(input_size_1, input_size_2, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print('model_path: {}'.format(model_path))
    return model


"""
Generate test file and anomaly line labels
"""


def generateTestFile(name, window_length):
    log_keys_sequences = list()
    abnormal_label = list()
    file = pandas.read_csv(name)
    k = 0
    for i in range(len(file)):
        line = [int(id) for id in file["Sequence"][i].strip().split(' ')]
        label = file["label"][i]
        if len(line) < window_length:
            continue
        log_keys_sequences.append(tuple(line))
        # print(label)
        if label == 1:
            abnormal_label.append(k)
        k += 1
    return log_keys_sequences, abnormal_label

def getLog(name, linenum):
    file = pandas.read_csv(name)
    return file["BlockId"][linenum]


"""
Return whether a block of event is abnormal or not
"""


def linePrediction_topK(predicted, label, num_candidates):
    dim0, dim1 = predicted.shape  # predicted is the output of all the windows in a log block
    abnormal_flag = 0
    for i in range(dim0):
        if label[i] not in torch.argsort(predicted[i])[
                           -num_candidates:]:  # The block is abnormal if the label is not in the top num_candidates
            abnormal_flag = 1
    return abnormal_flag


"""
In each log block, there are multiple windows, each window has a prediction result which is a probability.
If the prediction result of any window within a block is lower than the threshold, the block is abnormal.
"""


def linePrediction_Threshold(predicted, label, threshold):
    dim0, dim1 = predicted.shape  # predicted is the output of all the windows in a log block
    abnormal_flag = 0
    for i in range(dim0):
        # print(label[i], predicted[i][label[i]])
        if predicted[i][label[i]] < threshold:
            abnormal_flag = 1
    return abnormal_flag


"""The general idea is that since the label is attached to each block (each line), the window(length=5) is 
moved down each line. As the window moves down each line, if the predicted result doesn't match the ground 
truth in any of these windows, this block (this line) is flagged as abnormal. """


def do_predict(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
               model_output_directory, test_file, pattern_vec_file, num_candidates, threshold, test_batch_size=512):
    model = load_model(input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                       model_output_directory)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    result_str = ''
    abnormal_num = 0

    with open(pattern_vec_file, 'r') as pattern_file:
        PF = json.load(pattern_file)
        pattern_vec = {}

        # Cast each log event to its pattern vector
        i = 0
        for key, value in PF.items():
            pattern, vec = key, value
            pattern_vec[int(pattern)] = vec
            i = i + 1

    test_file_loader, abnormal_label = generateTestFile(test_file, window_length)  # Load test file

    start_time = time.time()
    print('Start Prediction')
    with torch.no_grad():
        batch_num = 0
        abnormal_flag = 0
        lineNum = 0
        n = 0
        p = 0
        # Batch with full length
        while n < (len(test_file_loader) - len(test_file_loader) % test_batch_size):
            batch_input_sequential = []
            batch_input_quantitative = []
            batch_label = []
            line_windowNum = []
            #  Each batch has (test_batch_size) log blocks
            for x in range(n, n + test_batch_size):
                line = test_file_loader[x]
                # Each line represents a block of log event
                i = 0
                # Skip the lines that are too short
                if len(line) < window_length:
                    continue
                # Slide the window in each line (window_length=5)
                while i < len(line) - window_length:
                    window_input_sequential = []
                    window_input_quantitative = []
                    for j in range(i, i + window_length):
                        # For sequential pattern, each log event in the window is cast to its pattern vector which eventually leads to a shape of 5*300
                        window_input_sequential.append(pattern_vec[line[i]])

                        quantitative_subwindow = [0] * num_of_classes  # Initiate quantitative input window
                        # For quantitative pattern, the quantitative window is used to generate the count vector (shape=5*31)
                        if j >= window_length:  # Full length windows
                            for k in range(j - window_length + 1, j + 1):
                                quantitative_subwindow[line[k]] += 1
                            window_input_quantitative.append(quantitative_subwindow)
                        else:  # Partial length windows
                            for m in range(0, j + 1):
                                quantitative_subwindow[line[m]] += 1
                            window_input_quantitative.append(quantitative_subwindow)

                    # The label is the index of the next log event
                    batch_label.append(line[i + window_length])
                    batch_input_sequential.append(window_input_sequential)
                    batch_input_quantitative.append(window_input_quantitative)
                    i += 1
                line_windowNum.append(i)

            seq = batch_input_sequential
            quan = batch_input_quantitative

            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_sequential).to(device)
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_quantitive).to(device)
            # print(seq.shape, quan.shape)
            test_output = model(seq, quan)
            # print(test_output.shape)
            #  Reconstruct the output to the original log blocks
            current_window_num = 0
            for k in range(
                    len(line_windowNum)):  # Reconstruct every line in a batch, each line has line_windowNum[k] windows
                line_label = []
                num_of_windows = line_windowNum[k]
                line_output = torch.empty(num_of_windows, num_of_classes)

                for i in range(current_window_num, current_window_num + num_of_windows):
                    line_output[i - current_window_num] = test_output[i]
                    line_label.append(batch_label[i])

                # Determine whether this line is abnormal or not.
                abnormal_flag = linePrediction_Threshold(line_output, line_label, threshold)
                #abnormal_flag = linePrediction_topK(line_output, line_label, num_candidates)
                if lineNum in abnormal_label:
                    ground_truth = 1
                else:
                    ground_truth = 0

                #print("line:", lineNum, "Predicted Label:", abnormal_flag, "Ground Truth:", ground_truth)

                # When this line(block) is flagged as abnormal
                if abnormal_flag == 1:
                    result_str += getLog(test_file, lineNum) + "\n"
                    abnormal_num += 1
                    if lineNum in abnormal_label:
                        TP += 1
                    else:
                        FP += 1

                # When this line(block) is not flagged as abnormal
                else:
                    if lineNum in abnormal_label:
                        FN += 1
                    else:
                        TN += 1
                lineNum += 1
                ALL += 1
                current_window_num += num_of_windows
                abnormal_flag = 0
                # End of for loop. Move on to the next line (Next block of log events)
            batch_num += 1
            n += test_batch_size

            # End of while loop. Deal with the remaining part.
        if n >= (len(test_file_loader) - len(test_file_loader) % test_batch_size):
            batch_input_sequential = []
            batch_input_quantitative = []
            batch_label = []
            line_windowNum = []
            # Deal with the remaining part
            for y in range(n, len(test_file_loader)):
                line = test_file_loader[y]
                # Each line represents a block of log event
                i = 0
                # Skip the lines that are too short
                if len(line) < window_length:
                    continue
                # Slide the window in each line (window_length=5)
                while i < len(line) - window_length:
                    window_input_sequential = []
                    window_input_quantitative = []
                    for j in range(i, i + window_length):
                        # For sequential pattern, each log event in the window is cast to its pattern vector which eventually leads to a shape of 5*300
                        window_input_sequential.append(pattern_vec[line[i]])

                        quantitative_subwindow = [0] * num_of_classes  # Initiate quantitative input window
                        # For quantitative pattern, the quantitative window is used to generate the count vector (shape=5*31)
                        if j >= window_length:  # Full length windows
                            for k in range(j - window_length + 1, j + 1):
                                quantitative_subwindow[line[k]] += 1
                            window_input_quantitative.append(quantitative_subwindow)
                        else:  # Partial length windows
                            for m in range(0, j + 1):
                                quantitative_subwindow[line[m]] += 1
                            window_input_quantitative.append(quantitative_subwindow)

                    # The label is the index of the next log event
                    batch_label.append(line[i + window_length])
                    batch_input_sequential.append(window_input_sequential)
                    batch_input_quantitative.append(window_input_quantitative)
                    i += 1
                line_windowNum.append(i)

            seq = batch_input_sequential
            quan = batch_input_quantitative

            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_sequential).to(device)
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_quantitive).to(device)
            # print(seq.shape, quan.shape)
            test_output = model(seq, quan)
            # print(test_output.shape)

            current_window_num = 0
            for k in range(len(line_windowNum)):
                line_label = []
                num_of_windows = line_windowNum[k]
                line_output = torch.empty(num_of_windows, num_of_classes)

                for i in range(current_window_num, current_window_num + num_of_windows):
                    line_output[i - current_window_num] = test_output[i]
                    line_label.append(batch_label[i])

                # Determine whether this line is abnormal or not.
                abnormal_flag = linePrediction_Threshold(line_output, line_label, threshold)
                #abnormal_flag = linePrediction_topK(line_output, line_label, num_candidates)
                if lineNum in abnormal_label:
                    ground_truth = 1
                else:
                    ground_truth = 0

                #print("line:", lineNum, "Predicted Label:", abnormal_flag, "Ground Truth:", ground_truth)

                # When this line(block) is flagged as abnormal
                if abnormal_flag == 1:
                    result_str += getLog(test_file, lineNum) + "\n"
                    abnormal_num += 1
                    if lineNum in abnormal_label:
                        TP += 1
                    else:
                        FP += 1

                # When this line(block) is not flagged as abnormal
                else:
                    if lineNum in abnormal_label:
                        FN += 1
                    else:
                        TN += 1
                lineNum += 1
                ALL += 1
                current_window_num += num_of_windows
                abnormal_flag = 0
                # End of for loop. Move on to the next line (Next block of log events)

    # Compute precision, recall and F1-measure
    if TP + FP == 0:
        P = 0
    else:
        P = 100 * TP / (TP + FP)

    if TP + FN == 0:
        R = 0
    else:
        R = 100 * TP / (TP + FN)

    if P + R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)

    Acc = (TP + TN) * 100 / ALL
    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

    result_dict=dict()
    result_dict["Number of Abnormal Log Blocks\t"] = str(format(abnormal_num))
    result_dict["Test Precision\t"] = str(format(P,".3f"))+"%"
    result_dict["Test Recall\t"] = str(format(R,".3f"))+"%"
    result_dict["Test F-Score\t"] = str(format(F1,".3f"))+"%"
    result_dict["Test Accuracy\t"]=str(format(Acc,".3f"))+"%"
    #result_dict["Elapsed Time\t"] = str(format(elapsed_time)) + "%"

    #result_str = 'FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN)

    return result_str, result_dict

