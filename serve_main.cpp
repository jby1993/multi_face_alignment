#include "cnndensefeature.h"
#include "io_utils.h"
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "train.h"
int main(int argc, char *argv[])
{

    train my_train;
    my_train.set_train_data_root("../../multi_learn_data/clip_data_224/");
    my_train.set_save_model_root("../save_model/");
    my_train.read_train_img_datas("../resource/debug_meshpara_list.txt","../resource/debug_permesh_imglists.txt");
    my_train.train_model();



    std::cout<<"done"<<std::endl;
    return 0;
}
