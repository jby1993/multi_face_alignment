#ifndef TRAIN2_H
#define TRAIN2_H
#include <list>
#include "face_imgs.h"
#include "part_3dmm_face.h"
#include "cnndensefeature.h"

class train2
{
    typedef part_3DMM_face Face;
public:
    train2(int thread_num);
    void read_img_datas(const std::string &meshpara_list, const std::string &permesh_imglist);
    void train_model();
    void verify_model();
    void save_verify_result(const std::string& root);
    void read_trained_model(const std::string &root, int casscade_num);
    void set_data_root(const std::string &root){m_data_root=root;}
    void set_save_model_root(const std::string &root){m_savemodel_root=root;}
    void set_feature_compute_gpu(const std::vector<int> ids);
    void set_matrix_compute_gpu(int id){m_gpuid_for_matrix_compute=id;}
    void set_feature_compute_thread_num(int num){m_threadnum_for_compute_features=num;}
private:
    void initial_shape_exp_with_mean();
    void initial_para();
    void compute_groundtruth_keypos_multithread();
    void compute_keypos_visibility();
    void read_para_mean_sd();
    void compute_shapes_exps_mean_sd();
    void compute_keypoint_visible_multi_thread(const Eigen::MatrixXf &R, const std::vector<int> &ids,std::vector<bool> &visuals, int thread_id);
    void compute_delta_para(Eigen::MatrixXf &delta_para);
    void compute_delta_shape_exp(Eigen::MatrixXf &delta_para);
    void show_delta_para_info();
    void show_delta_keypos_info();
    void show_delta_shape_exp_info();
    void compute_all_visible_features_multi_thread();
    void compute_update_keypos_R();
    void compute_update_para_R(bool use_U);
    void save_keypos_R(int casscade_level);
    void save_paras_R(int casscade_level);
    void optimize_all_shape_exp();

    void update_keypos_R();
    void update_para_R(bool use_U);

    //solve ||x-Rf||^2+lamda*||R||
    void compute_R(const MatrixXf &x, const MatrixXf &f, float lamda, MatrixXf &R);
    void save_R(const MatrixXf &R, const std::string &file_name);
    bool check_matrix_invalid(const Eigen::MatrixXf &matrix);
    void read_keypos_R(const std::string &readmodel_root, int casscade_level);
    void read_para_R(const std::string &readmodel_root, int casscade_level);
    void read_R(MatrixXf &R, const std::string &file_name);
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
    int m_threadnum_for_compute_features;
    std::vector<int> m_gpuid_for_feature_computes;
    int m_gpuid_for_matrix_compute; //mainly for rankUpdate computation
#ifdef USE_CNNFEATURE
    std::vector<CNNDenseFeature> m_feature_detectors;
#else

#endif
    std::vector<part_3DMM_face>  m_3dmm_meshs;
    std::string m_data_root;
    std::string m_savemodel_root;
    std::string m_para_mean_sd_file;

    Eigen::MatrixXf m_train_keypos; //left top origine
    Eigen::MatrixXi m_keypos_visible;   //keypoints visibility
    Eigen::MatrixXf m_train_paras;
    Eigen::MatrixXf m_train_shapes;
    Eigen::MatrixXf m_train_exps;
    Eigen::MatrixXf m_visible_features;

    Eigen::MatrixXf m_groundtruth_keypos;
    Eigen::VectorXf m_groundtruth_paras_sd;
    Eigen::VectorXf m_groundtruth_paras_mean;
    Eigen::VectorXf m_groundtruth_shapes_exps_sd;
    std::vector<Eigen::MatrixXf> m_keypos_Rs;
    std::vector<Eigen::MatrixXf> m_para_Rs;
};

#endif // TRAIN2_H
