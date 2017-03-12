#ifndef FACE_IMG_H
#define FACE_IMG_H
#include <string>
#include <Eigen/Dense>
//all train img are normalized, 224*224, left top origin, row major
//this is used for one img train without mesh ground truth, not multi imgs.
class face_img
{
public:
    face_img(int paranum, int keyposnum, const std::string &name);
    void read_img(const std::string &imgfile);
    void read_para(const std::string &posefile);
    void read_land(const std::string &landfile);
    int img_length(){return m_img_length;}
    const Eigen::VectorXf &get_groundtruth_para(){return m_groundtruth_para;}
    const Eigen::VectorXf &get_groundtruth_land(){return m_groundtruth_land;}
    const std::vector<float> &get_img_data(){return m_img_data;}
    const std::string &get_face_name(){return m_face_name;}
private:
    std::string m_face_name;
    int m_img_length;
    int m_para_num;
    int m_keypos_num;
    Eigen::VectorXf m_groundtruth_para;
    Eigen::VectorXf m_groundtruth_land;
    std::vector<float>    m_img_data;
};

#endif // FACE_IMG_H
