#ifndef CNNDENSEFEATURE_H
#define CNNDENSEFEATURE_H
#include <string>
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include <Eigen/Dense>
using namespace caffe;
class CNNDenseFeature
{
public:
    CNNDenseFeature();
    //input image must be normalized image, i.e 224*224*float, left up origin, row major
    void set_data(const std::vector<float> &data);
    const std::vector<float>& get_features();
    void get_compute_visible_posfeatures(const Eigen::MatrixXf &pos, const std::vector<bool> &visible, Eigen::VectorXf &visible_features);
    //void get_features(const std::vector<bool> &visibles,const Eigen::Matrix2Xf &points_pos, Eigen::VectorXf &descriptors);
private:
    void construct_net(std::string model, std::string weights);
    void feature_compute();
private:
    boost::shared_ptr<Net<float> > feature_net;
    std::vector<float> mean_image;
    std::vector<float> features;
    int f_length;
    int f_dimension;
    int img_size;
    bool is_updated;
};

#endif // CNNDENSEFEATURE_H
