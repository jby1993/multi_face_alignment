#ifndef TRAIN_H
#define TRAIN_H
#include <list>
#include "face_imgs.h"
#include "part_3dmm_face.h"
#include "cnndensefeature.h"
#include "divide_orientation_space.h"
#include "random_tool.h"
#define USE_CNNFEATURE
class train
{
    typedef part_3DMM_face Face;
public:
    train(int thread_num);
    ~train(){}
    void read_train_img_datas(const std::string &meshpara_list, const std::string &permesh_imglist);
    void train_model();
    void save_para_result(int casscade_level);
    void save_shape_exp_result(int casscade_level);
    void set_train_data_root(const std::string &root){m_traindata_root=root;}
    void set_save_model_root(const std::string &root){m_savemodel_root=root;}
    void set_feature_compute_gpu(int id){m_gpuid_for_feature_compute=id;}
    void set_matrix_compute_gpu(int id){m_gpuid_for_matrix_compute=id;}
    void set_feature_compute_thread_num(int num){m_threadnum_for_compute_features=num;}

    //compute C=a*A*AT, C can not pre allocate
    static void my_gpu_rankUpdated(Eigen::MatrixXf &C, const Eigen::MatrixXf &A, float a, int gpu_id);
private:
    void initial_shape_exp_with_mean();
    void initial_para();
    void compute_para_mean_sd();
    void compute_shapes_exps_mean_sd();
    //compute groundtruth para substract train para
    void compute_delta_para(Eigen::MatrixXf &delta_para);
    void compute_delta_shape_exp(Eigen::MatrixXf &delta_para);
//    void compute_all_visible_features();
    void compute_all_visible_features_multi_thread();
//    void compute_per_face_imgs_visiblefeatures(const std::list<face_imgs>::iterator &iter, int index, int img_size);
    void compute_per_face_imgs_visiblefeatures_multi_thread(face_imgs *iter, int index, int img_size, int thread_id);
    void compute_keypoint_visible_multi_thread(const MatrixXf &R, std::vector<bool> &visuals, int thread_id);
    void compute_paras_R_b();
    void update_para();
    void show_delta_para_info();

    void compute_regular_feature_from_multi_imgs(int before_img_size, const std::vector<int> &choose_imgs_ids, MatrixXf &regu_features);
    void compute_shapes_exps_R_b();
    void update_shape_exp();
    void show_delta_shape_exp_info();

    void expand_delta_x_with_num(int num, const Eigen::MatrixXf &delta_x, Eigen::MatrixXf &expand_delta_x);
    void depand_delta_x_with_num(int num, const Eigen::MatrixXf &expand_delta_x, Eigen::MatrixXf &delta_x);

    bool check_matrix_invalid(const Eigen::MatrixXf &matrix);


private:
    //using list to allocate non continus memory, when data is large, which is more safe
    std::list<face_imgs>    m_face_imgs;
    //the 2 is for multithread need
    std::vector<face_imgs*> m_face_imgs_pointer;    //save m_face_imgs pointer with list iterator order
    std::vector<int> m_before_imgsize;  //save before one face_imgs all img num

    int m_face_img_num;
    int m_all_img_num;
    int m_para_num;
    int m_feature_size;
    int m_casscade_sum;
    int m_casscade_level;
    int per_face_img_random_train_data_num;
    int m_threadnum_for_compute_features;
    int m_gpuid_for_feature_compute;
    int m_gpuid_for_matrix_compute; //mainly for rankUpdate computation
#ifdef USE_CNNFEATURE
    std::vector<CNNDenseFeature> m_feature_detectors;
#else

#endif
    std::vector<part_3DMM_face>  m_3dmm_meshs;
    std::string m_traindata_root;
    std::string m_savemodel_root;
    divide_orientation_space m_orien_choose;
    random_tool m_random_tool;

    Eigen::MatrixXf m_train_paras;
    Eigen::MatrixXf m_train_shapes;
    Eigen::MatrixXf m_train_exps;
    Eigen::MatrixXf m_visible_features;
    Eigen::MatrixXf m_regu_features;

    Eigen::VectorXf m_groundtruth_paras_sd;
    Eigen::VectorXf m_groundtruth_paras_mean;
    Eigen::VectorXf m_groundtruth_shapes_exps_sd;
    std::vector<Eigen::MatrixXf> m_para_Rs;
    std::vector<Eigen::VectorXf> m_para_bs;
    //add b into R, so donot need b
    std::vector<Eigen::MatrixXf > m_shape_exp_Rs;
};

#endif // TRAIN_H
