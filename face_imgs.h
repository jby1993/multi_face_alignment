#ifndef FACE_IMGS_H
#define FACE_IMGS_H
#include <string>
#include <vector>
#include <Eigen/Dense>
//all train img are normalized, 224*224, left top origin, row major
class face_imgs
{
public:
    face_imgs(const std::string &root, const std::string &meshparas_name,int shape_dim, int exp_dim);
    void set_img_num(int num);
    void read_imgs();
    void read_pose_paras();
    void read_shape_exp();
    int img_size(){return m_img_num;}
    int img_length(){return m_img_length;}
    Eigen::VectorXf get_groundtruth_para(int id){return m_groundtruth_paras.col(id);}
    const Eigen::MatrixXf& get_groundtruth_paras(){return m_groundtruth_paras;}
    const Eigen::VectorXf& get_groundtruth_shapes(){return m_groundtruth_shapes;}
    const Eigen::VectorXf& get_groundtruth_exps(){return m_groundtruth_exps;}
    const std::vector<float>& get_img(int id){return m_imgs_data[id];}
private:
    void initial();
    void resize_data();
private:
    std::string m_face_name;
    int m_img_num;
    int m_shape_dimension;
    int m_exp_dimension;
    std::string m_files_root;
    int m_img_length;
    //img datas are left up origin, to make correspond to CNNFeature
    std::vector<std::vector<float> >    m_imgs_data;
    //all this paras's 3DMM is left down origin
    Eigen::MatrixXf m_groundtruth_paras;
    Eigen::VectorXf m_groundtruth_shapes;
    Eigen::VectorXf m_groundtruth_exps;
};

#endif // FACE_IMGS_H
