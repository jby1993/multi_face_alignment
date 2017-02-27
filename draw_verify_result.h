#ifndef DRAW_VERIFY_RESULT_H
#define DRAW_VERIFY_RESULT_H
#include <string>
#include <vector>
#include "part_3dmm_face.h"
class draw_verify_result
{
    typedef part_3DMM_face Face;
public:
    draw_verify_result(const std::string &data_root, const std::string &result_root, int threads_num);
    void draw(bool is_draw_dense=false, bool is_draw_unvisible=true);
private:
    void read_result_file_names(std::vector<std::string> &meshfiles, std::vector<std::vector<std::string> > &per_mesh_pose_files);
    void compute_keypoint_visible_multi_thread(const Eigen::MatrixXf &R, const std::vector<int> &ids,std::vector<bool> &visuals, int thread_id);
private:
    int m_threads_num;
    std::string m_data_root;
    std::string m_result_root;
    std::vector<Face> m_3dmm_meshs;
};

#endif // DRAW_VERIFY_RESULT_H
