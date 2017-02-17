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
    train();
    ~train(){}
    void read_train_img_datas(const std::string &meshpara_list, const std::string &permesh_imglist);
    void train_model();
    void save_para_result(int casscade_level);
    void save_shape_exp_result(int casscade_level);
private:
    void initial_shape_exp_with_mean();
    void initial_para();
    void compute_para_mean_sd();
    void compute_shapes_exps_mean_sd();
    //compute groundtruth para substract train para
    void compute_delta_para(Eigen::MatrixXf &delta_para);
    void compute_delta_shape_exp(Eigen::MatrixXf &delta_para);
    void compute_all_visible_features();
    void compute_per_face_imgs_visiblefeatures(const std::list<face_imgs>::iterator &iter, int index, int img_size);
    void compute_keypoint_visible(const MatrixXf &R, int width, int height, std::vector<bool> &visuals);
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
    int m_face_img_num;
    int m_all_img_num;
    int m_para_num;
    int m_feature_size;
    int m_casscade_sum;
    int m_casscade_level;
    int per_face_img_random_train_data_num;
#ifdef USE_CNNFEATURE
    CNNDenseFeature m_feature_detector;
#else

#endif
    part_3DMM_face  m_3dmm_mesh;
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
