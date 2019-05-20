import os
import time

cur_path = os.path.dirname(os.getcwd())  # image_and_anno dir
img_dir = cur_path + "/images_val"
original_annofile = "wider_face_val_bbx_gt.txt"
gen_annofile = "gen_anno_file_val"
gen_enable = False

if gen_enable:
    print("start generating")
    t1 = time.time()
    with open(original_annofile, 'r') as f:
        with open(gen_annofile, 'w') as target_f:
            current_line = f.readline()
            while current_line:
                if current_line.endswith(".jpg\n"):  # one image begins**************
                    img_path = img_dir + "/" + current_line[:-1]  # remove \n

                    current_line = f.readline()
                    box_num = int(current_line)
                    box_list = []
                    for i in range(box_num):  # one box begins****************
                        current_line = f.readline()
                        box_info = current_line.split(" ")
                        box_list += box_info[:4]
                    # write to file
                    target_f.writelines(img_path+" "+str(box_num)+" "+" ".join(box_list)+"\n")
                else:
                    print("error occurs!")
                    break

                current_line = f.readline()

            target_f.close()
        f.close()
    t2 = time.time()
    print("generating done, cost {:.3f}s".format(t2-t1))

print("gen_file test begins")
with open(gen_annofile, "r") as f:
    current_line = f.readline()
    line_num = 1
    while current_line:
        line = current_line.split(" ")
        if line[0].endswith(".jpg") and int(line[1]) >= 1 and not [int(i) for i in line[2:6]] == [0, 0, 0, 0]:
            if (len(line)-2) % 4 != 0:
                print("%4 error in {}".format(line_num))
            else:
                pass
        else:
            print("100 error in {}".format(line_num))

        current_line = f.readline()
        line_num += 1

    f.close()
print("test finished")
