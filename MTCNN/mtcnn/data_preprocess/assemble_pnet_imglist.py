"""
    assemble all anno files and choose samples randomly
    input:
        postive/part/negative anno file (\anno_store\xxx_12.txt)
    output:
        overall anno file (\anno_store\imglist_anno_12)
"""
import os
import sys
import mtcnn.data_preprocess.assemble as assemble
projdir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

pnet_postive_file = projdir + r'\anno_store\pos_12.txt'
pnet_part_file = projdir + r'\anno_store\part_12.txt'
pnet_neg_file = projdir + r'\anno_store\neg_12.txt'
pnet_landmark_file = projdir + r'\anno_store\landmark_12.txt'
imglist_filename = projdir + r'\anno_store\imglist_anno_12.txt'

if __name__ == '__main__':

    anno_list = [pnet_postive_file, pnet_part_file, pnet_neg_file]

    # anno_list.append(pnet_landmark_file)

    chose_count = assemble.assemble_data(imglist_filename, anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)
