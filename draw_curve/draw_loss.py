import numpy as np
from draw_curve.draw_curves import draw_curve, assemble_data


if __name__ == "__main__":
    fliter = 30
    idx = 0
    data_files1_0 = ["../train_net2net/weights_of_mine/try1/intermedia_loss_" + str(iter_num) + "000.npy"
                     for iter_num in range(2, 105, 2)]
    data1_0 = assemble_data(data_files1_0, False, flit_num=fliter)
    data_files2_0 = ["../train_net2net/weights_of_mine/try2/intermedia_loss_" + str(iter_num) + "000.npy"
                     for iter_num in range(2, 47, 2)]
    data2_0 = assemble_data(data_files2_0, False)

    data_files1_1 = ["../weights_of_mine/try1_from104000/Mobile_train_loss_"+str(iter_num)+ "000.npy"
                     for iter_num in range(2, 19, 2)]
    data_files1_1_eval = ["../weights_of_mine/try1_from104000/Mobile_eval_loss_"+str(iter_num)+ "000.npy"
                          for iter_num in range(2, 19, 2)]
    data1_1 = assemble_data(data_files1_1, index=idx, flit_num=fliter)
    data1_1_eval = assemble_data(data_files1_1_eval, iseval=True, isoverall=False)

    data_files3 = ["../weights_of_mine/try3/Mobile_train_loss_" + str(iter_num) + "000.npy"
                   for iter_num in range(3, 25, 3)]
    data3 = assemble_data(data_files3, index=idx, flit_num=fliter)
    data_files3_eval = ["../weights_of_mine/try3/Mobile_eval_loss_"+str(iter_num)+'000.npy'
                        for iter_num in range(3, 25, 3)]
    data3_eval = assemble_data(data_files3_eval, iseval=True, isoverall=False)

    data_files_repo0 = ("../weights_of_mine/repo_Res50/Res50_loss_3000.npy",
                        "../weights_of_mine/repo_Res50/Res50_loss_6000.npy",
                        "../weights_of_mine/repo_Res50/Res50_loss_9000.npy",
                        "../weights_of_mine/repo_Res50/Res50_loss_12000.npy",
                        "../weights_of_mine/repo_Res50/Res50_loss_15000.npy"
                        )
    data_repo0 = assemble_data(data_files_repo0, index=idx, flit_num=fliter)

    # draw_curve((data_repo0, data1_1, data3),
    #            ('repo0', 'try1', 'try3'),
    #            "loss", xlabel="epochs", ylabel="loss")
    # draw_curve((data1_0, ),
    #            ('', ),
    #            "", xlabel="epochs", ylabel="loss")
    # draw_curve((data2_1, data1_2, data1_1),
    #            ('epochs=0', 'epochs=36000', 'epochs=104000'),
    #            "", xlabel="epochs", ylabel="loss")

    draw_curve((data3, data3_eval),
               ('train loss', 'evaluate loss'),
               "method 2", xlabel="epochs", ylabel="loss")
    draw_curve((data1_1, data1_1_eval),
               ('train loss', 'evaluate loss'),
               "method 1", xlabel="epochs", ylabel="loss")
    draw_curve((data_repo0, ),
               ('train loss',),
               "reproduce", xlabel="epochs", ylabel="loss")

    # draw_curve((data3, data1_1, data3_eval, data1_1_eval, data_repo0),
    #            ('train loss for method 1', 'train_loss for method 2',
    #             'evaluate loss for method1', 'evaluate loss for method2',
    #             'train_loss for repoduce'),
    #            "", xlabel='epochs', ylabel='loss')


