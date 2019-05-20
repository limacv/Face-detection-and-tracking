import os
import numpy as np


def assemble_data(output_file, anno_file_list):
    """
    assemble the annotations to one file
    :input
        anno_file_list
    :output
        assembled file (output_file)
    :return
        numbers of items choosed
    """

    if len(anno_file_list) == 0:
        return 0

    if os.path.exists(output_file):
        os.remove(output_file)

    # 对于每一个文件
    for anno_file in anno_file_list:
        with open(anno_file, 'r') as f:
            print(anno_file)
            anno_lines = f.readlines()

        base_num = 250000
        # generate samples from anno_lines
        if len(anno_lines) > base_num * 3:  # 如果太多了取其中的750000
            idx_keep = np.random.choice(len(anno_lines), size=base_num * 3)
        # elif len(anno_lines) > 100000:  # 如果不怎么多就全要
        else:
            idx_keep = np.random.choice(len(anno_lines), size=len(anno_lines))
        # else:
        #     idx_keep = np.arange(len(anno_lines))
        #     np.random.shuffle(idx_keep)
        chose_count = 0

        # write output file
        with open(output_file, 'a+') as f:
            for idx in idx_keep:
                # write lables of pos, neg, part images
                f.write(anno_lines[idx])
                chose_count += 1

    return chose_count
