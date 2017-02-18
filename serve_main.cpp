#include "cnndensefeature.h"
#include "io_utils.h"
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "train.h"
int main(int argc, char *argv[])
{

    //close caffe log output
    FLAGS_minloglevel=1;
    train my_train;
    my_train.set_feature_compute_gpu(0);
    my_train.set_feature_compute_thread_num(1);
    my_train.set_matrix_compute_gpu(0);
    my_train.set_train_data_root("/data_b/jby/multiclassdata/clip_data_224/");
    my_train.set_save_model_root("../save_model/");
    my_train.read_train_img_datas("../resource/train_meshpara_list.txt","../resource/train_permesh_imglists.txt");
    my_train.train_model();



    std::cout<<"done"<<std::endl;
    return 0;
}
