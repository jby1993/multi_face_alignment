#ifndef PART_3DMM_FACE_H
#define PART_3DMM_FACE_H
#include "tri_mesh.h"
#include <Eigen/Dense>
using namespace Eigen;

class part_3DMM_face
{
public:

public:
    part_3DMM_face();
    void set_shape(const VectorXf &shape);
    void set_exp(const VectorXf &exp);    
    void get_vertex_matrix(MatrixXf &verts, bool is_whole);
    const TriMesh& get_part_face();
    void get_mean_normal(int v_id, TriMesh::Normal &mean_normal, int neighbor_size=5, bool is_whole=false);


    static int get_shape_pcanum();
    static int get_exp_pcanum();
    static int get_keypoints_size();
    static int get_dense_keypoint_size();
    static const std::vector<int>& get_part_keypoints();
    static int get_dense_keypoint_id(int i);
    static const std::vector<int>& get_dense_keypoint();
    static const std::vector<int>& get_partv2wholev();
    static const Eigen::MatrixXf& get_shape_st();
    static const Eigen::MatrixXf& get_exp_st();
    //ids are part face id, first convert to whole face, then extract base
    static void get_mean_vertex_base_on_ids(const std::vector<int> &ids,Eigen::MatrixXf &base);
    static void get_shape_vertex_base_on_ids(const std::vector<int> &ids, Eigen::MatrixXf &base);
    static void get_exp_vertex_base_on_ids(const std::vector<int> &ids, Eigen::MatrixXf &base);
    static void get_vertex_base_on_ids(const std::vector<int> &ids, Eigen::MatrixXf &base);
private:
    void initial();
    void update_mesh(bool is_whole);
private:
    TriMesh m_whole_face;
    TriMesh m_part_face;
    bool is_whole_updated;
    bool is_part_updated;
    VectorXf m_shape;
    VectorXf m_exp;


};

#endif // PART_3DMM_FACE_H
