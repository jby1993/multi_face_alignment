#ifndef WHOLE_3DMM_FACE_H
#define WHOLE_3DMM_FACE_H
#include "tri_mesh.h"
#include <Eigen/Dense>
//this is zhu's 3dmm model, only has whole model
using namespace Eigen;
class whole_3dmm_face
{
public:
    whole_3dmm_face();
    void set_shape(const VectorXf &shape);
    void set_exp(const VectorXf &exp);
    void set_shape_exp(const std::vector<float> &para);
    void get_vertex_matrix(MatrixXf &verts);
    const TriMesh& get_mesh();
    void get_mean_normal(int v_id, TriMesh::Normal &mean_normal, int neighbor_size=3);
    void get_neighborIds_around_v(int v_id, std::vector<int> &ids, int neighbor_size = 10);

    static int get_shape_pcanum();
    static int get_exp_pcanum();
    static int get_keypoints_size();
    static int get_featurekeypoints_size();
    static const std::vector<int>& get_keypoints();
    static const std::vector<int>& get_featurekeypoints();
    static int get_keypoint_id(int i);
    static int get_featurekeypoint_id(int i);
    static const Eigen::MatrixXf& get_shape_st();
    static const Eigen::MatrixXf& get_exp_st();
    //ids are part face id, first convert to whole face, then extract base
    static void get_mean_vertex_base_on_ids(const std::vector<int> &ids,Eigen::MatrixXf &base);
    static void get_shape_vertex_base_on_ids(const std::vector<int> &ids, Eigen::MatrixXf &base);
    static void get_exp_vertex_base_on_ids(const std::vector<int> &ids, Eigen::MatrixXf &base);
    static void get_vertex_base_on_ids(const std::vector<int> &ids, Eigen::MatrixXf &base);


private:
    void initial();
    void update_mesh();
private:
    TriMesh m_mesh;
    bool is_updated;
    Eigen::VectorXf m_shape;
    Eigen::VectorXf m_exp;
};

#endif // WHOLE_3DMM_FACE_H
