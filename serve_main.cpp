#include "cnndensefeature.h"
#include "io_utils.h"
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "train.h"
#include "train2.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"

DEFINE_string(phase, "TRAIN",
    "Essential; network phase (TRAIN or VERIFY). must have one.");
DEFINE_string(read_model_root, "",
    "Optional; if phase is not TRAIN, trained models are needed.");
DEFINE_int32(read_casscade_num,5,
    "Optional; if need trained models, give read casscade num.");
DEFINE_string(feature_compute_gpus, "0",
    "Optional; compute feature in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. ");
DEFINE_int32(matrix_compute_gpu, 0,
    "Optional; matrix compute use GPU id.");

DEFINE_int32(feature_threadnum, 1,
    "Optional; feature compute use thread num. per thread correspond to a GPU,"
             "so must equal to feature_compute_gpus num");
DEFINE_string(data_root, "",
    "Essential; data save root, must have / end.");
DEFINE_string(save_root, "",
    "Essential; save model root, must have / end.");
DEFINE_string(meshpara_list, "",
    "Essential; read meshpara file.");
DEFINE_string(permesh_imglists, "",
    "Essential; read permesh imglists file.");

void test_cnnfeature()
{
    //    CNNDenseFeature feat;
    //    cv::Mat tmp=cv::imread("../lfpw_12_0.jpg",cv::IMREAD_GRAYSCALE);
    //    tmp.convertTo(tmp,CV_32F,1.0);

    //    std::vector<float> img;
    //    img.resize(tmp.rows*tmp.cols,0.0);
    //    float *now_data = img.data();
    //    for(int i=0; i<tmp.rows; i++)
    //    {
    //        const float *ptr = tmp.ptr<float>(i);
    //        memcpy(now_data, ptr, sizeof(float)*tmp.cols);
    //        now_data+=tmp.cols;
    //    }


    //    feat.set_data(img);
    ////    feat.feature_compute();
    //    std::cout<<"start compute feature"<<std::endl;
    //    io_utils::write_all_type_to_bin<float>(feat.get_features(),"test.bin");
}
void read_feature_gpu_ids(std::vector<int> &gpus)
{
    gpus.clear();
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    CHECK_GE(FLAGS_matrix_compute_gpu,0)<<"matrix compute gpu id set wrong";
    CHECK_LT(FLAGS_matrix_compute_gpu,count)<<"matrix compute gpu id set wrong";
    if (FLAGS_feature_compute_gpus == "all")
    {
      for (int i = 0; i < count; ++i) {
        gpus.push_back(i);
      }
    }
    else if (FLAGS_feature_compute_gpus.size())
    {
      std::vector<std::string> strings;
      boost::split(strings, FLAGS_feature_compute_gpus, boost::is_any_of(","));
      for (int i = 0; i < strings.size(); ++i)
      {
        int id = boost::lexical_cast<int>(strings[i]);
        CHECK_LT(id,count)<<"set feature gpu ids have some are wrong";
        gpus.push_back(id);
      }
      CHECK_LE(gpus.size(),count)<<"set feature gpu num is beyond the total GPUs num";
    }
    else
    {
      CHECK_EQ(gpus.size(), 0);
      LOG(FATAL)<<"feature_compute_gpus can not be NULL!";
    }
}

void call_meodule(train2 *compute)
{
    if (FLAGS_phase == "")
        LOG(FATAL)<<"phase must be \"TRAIN\" or \"VERIFY\"";
    else if (FLAGS_phase == "TRAIN")
        compute->train_model();
    else if (FLAGS_phase == "VERIFY")
    {
        CHECK_GT(FLAGS_read_model_root.size(),0)<<"VERIFY phase need a trained model";
        CHECK_GT(FLAGS_read_casscade_num,0)<<"VERIFY phase need a correct casscade num";
        compute->read_trained_model(FLAGS_read_model_root,FLAGS_read_casscade_num);
        compute->verify_model();
        compute->save_verify_result(FLAGS_save_root);
    }
    else
        LOG(FATAL) << "phase must be \"TRAIN\" or \"VERIFY\"";
}

int main(int argc, char *argv[])
{

    FLAGS_alsologtostderr = 1;
    //close caffe log output
//    FLAGS_minloglevel=1;
    // Google flags.
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Google logging.
    ::google::InitGoogleLogging(argv[0]);
    // Provide a backtrace on segfault.
    ::google::InstallFailureSignalHandler();

    std::vector<int> feature_compute_gpus;
    read_feature_gpu_ids(feature_compute_gpus);
    int thread_num = FLAGS_feature_threadnum;
    if(thread_num!=feature_compute_gpus.size())
    {
        LOG(WARNING)<<"feature compute thread does not match with feature compute gpus size,"
                   "change thread num.";
        thread_num = feature_compute_gpus.size();
    }
//    CHECK_EQ(FLAGS_feature_threadnum,feature_compute_gpus.size())<<"feature compute thread must have same size with feature compute gpus";
    LOG(INFO)<<"feature compute thread num"<<FLAGS_feature_threadnum;
    LOG(INFO)<<"feature compute use gpu ids:";
    for(int i=0;i<FLAGS_feature_threadnum;i++)
        LOG(INFO)<<feature_compute_gpus[i]<<",";

    train2 my_train(thread_num);
    my_train.set_feature_compute_gpu(feature_compute_gpus);
    my_train.set_matrix_compute_gpu(FLAGS_matrix_compute_gpu);
    my_train.set_data_root(FLAGS_data_root);
    my_train.set_save_model_root(FLAGS_save_root);
    my_train.read_img_datas(FLAGS_meshpara_list,FLAGS_permesh_imglists);
//    my_train.train_model();
    call_meodule(&my_train);



    LOG(INFO)<<"all done";
    return 0;
}

