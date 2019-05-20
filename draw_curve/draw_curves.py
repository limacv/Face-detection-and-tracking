import numpy as np
import matplotlib.pyplot as plt


def draw_curve(data_list, labels, title: str, xlabel, ylabel, label_position='tr'):
    """
    :param data_list:
            element of list: npArray
                            or (x_npArray, y_npArray) : x_npArray & y_npArray should have the same size
    """
    plt.figure()
    plt.title(title)
    for data, label in zip(data_list, labels):
        if isinstance(data, np.ndarray):
            plt.plot(data, label=label)
        elif isinstance(data, tuple):
            plt.plot(data[0], data[1], label=label)
    if label_position == 'bl':
        plt.legend(loc='lower left')
    elif label_position == 'br':
        plt.legend(loc='lower right')
    elif label_position == 'tl':
        plt.legend(loc='upper left')
    elif label_position == 'tr':
        plt.legend(loc='upper right')
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.grid()
    plt.show()


def assemble_data(data_files, isoverall=True, index=1,
                  flit_num=1, iseval=False, eval_freq=500):
    """
    :param data_files:
    :param isoverall:
    :param index:
        index = 0: total_loss
              = 1: face_loc_loss
              = 2: face_conf_loss
              = 3: head_loc_loss
              = 4: head_conf_loss
    :param flit_num:
        when flit_num <= 1, fliter is off
    :param iseval:
        if iseval=True, than return npArray will do linear interpolate
        according to eval_freq
    :param eval_freq:
    :return: -> npArray
    """
    assembled = np.array([])
    datas = (np.load(data_file) for data_file in data_files)
    for data in datas:
        data = data[data.nonzero()]
        if isoverall:
            assembled = np.hstack((assembled, data.reshape(5, -1)[index, :]))
        else:
            assembled = np.hstack((assembled, data))
    if flit_num > 1:
        temp = assembled[0]
        his_temp = 0
        for i, element in enumerate(assembled):
            his_temp += element
            if (i + 1) % flit_num == 0:
                temp = his_temp / flit_num
                his_temp = 0
            assembled[i] = temp
    if iseval:
        assembled = np.hstack((assembled.reshape(-1, 1), np.zeros(( assembled.shape[0], eval_freq-1))))
        for i, a in enumerate(assembled):
            if i == assembled.shape[0] - 1:
                a.fill(a[0])
                break
            for j in range(a.shape[0]):
                a[j] = (a[0] * (500 - j) + assembled[i+1, 0] * j) / 500
        assembled = assembled.reshape(-1)

    return assembled

