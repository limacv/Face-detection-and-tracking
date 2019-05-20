import numpy as np
from draw_curve.draw_curves import draw_curve


def gen_tp_fp(tf_conf):
    """
    generate the true_pos/false_pos array from the tf_conf

    :param tf_conf: npArray.shape(2, M)
    :return: true_pos, false_pos
        true_pos: npArray.shape(M)
        false_pos: npArray.shape(M)
    """
    _, M = tf_conf.shape
    true_pos, false_pos = np.zeros(M), np.zeros(M)
    for i in range(1, M+1):
        true_pos[i - 1] = np.count_nonzero(tf_conf[0, :i])
        false_pos[i - 1] = i - true_pos[i - 1]
    return true_pos, false_pos


# <<<<<<<<<<<<here write down file name in data_file_list>>>>>>>>>>>>>>>>>>>>>>>>>
pr_list, roc_list = [], []
data_file_list = ["./data/data_of_"+net_name+".npy"
                  for net_name in ('repo_my', 'try1', 'try3')]  # ('mtcnn', 'facebox', 'repo')]  # , ('repo_my', 'try1', 'try3')]
for data_file in data_file_list:
    data = np.load(data_file)

    truth_num = data[1, -1]
    tp, fp = gen_tp_fp(data[:, :-1])
    recall = tp/truth_num
    precision = tp / (tp + fp)
    pr_list.append((recall, precision))
    roc_list.append((fp, recall))


# <<<<<<<<<here change the labels' name cooresponding to data_file_list>>>>>>>>>>>>>>
labels = ('repoduce(ResNet50_original)', 'method 1(ResNet50_simplified)', 'method 2(SSDLite)')  # , 'repo_my', 'try1', 'try3')
draw_curve(pr_list, labels, "precision-recall curve",
           xlabel="recall", ylabel="precision", label_position='bl')

draw_curve(roc_list, labels, "roc curve",
           xlabel="true positive", ylabel='recall', label_position='br')
