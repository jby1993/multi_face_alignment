#include "whole_3dmm_face.h"
#include "io_utils.h"
#include "glog/logging.h"

static MatrixXf m_mean_shape;
static MatrixXf m_pca_shape;
static MatrixXf m_shape_st;
static MatrixXf m_mean_exp;
static MatrixXf m_pca_exp;
static MatrixXf m_exp_st;
static int m_shape_pcanum;
static int m_exp_pcanum;
static int m_whole_vnum;
static std::vector<int> m_whole_keypoints;
static bool is_3DMM_initialized=false;
whole_3dmm_face::whole_3dmm_face()
{
    initial();
}

void whole_3dmm_face::set_shape(const VectorXf &shape)
{
    m_shape=shape;
    is_updated=false;
}

void whole_3dmm_face::set_exp(const VectorXf &exp)
{
    m_exp = exp;
    is_updated=false;
}

void whole_3dmm_face::set_shape_exp(const std::vector<float> &para)
{
    LOG_IF(FATAL,para.size()!=m_shape_pcanum+m_exp_pcanum)<<"shape and exp para size wrong!";
    memcpy(m_shape.data(),para.data(),sizeof(float)*m_shape_pcanum);
    memcpy(m_exp.data(),para.data()+m_shape_pcanum,sizeof(float)*m_exp_pcanum);
    is_updated=false;
}

void whole_3dmm_face::get_vertex_matrix(MatrixXf &verts)
{
    if(!is_updated)
        update_mesh();
    verts.resize(3,m_mesh.n_vertices());
    TriMesh::VertexIter v_it = m_mesh.vertices_begin();
    const float *addrV = m_mesh.point(*v_it).data();
    memcpy(verts.data(),addrV,sizeof(float)*verts.size());
}

const TriMesh &whole_3dmm_face::get_mesh()
{
    if(!is_updated)
        update_mesh();
    return m_mesh;
}

void whole_3dmm_face::get_mean_normal(int v_id, TriMesh::Normal &mean_normal, int neighbor_size)
{
    if(!is_updated)   update_mesh();
    TriMesh *mesh=&m_mesh;
    std::set<int> neighbors;
    std::vector<int> ring;
    ring.push_back(v_id);
    for(int time=0;time<neighbor_size;time++)
    {
        std::vector<int> next_ring;
        for(int i=0;i<ring.size();i++)
        {
            if(neighbors.count(ring[i]))
                continue;
            neighbors.insert(ring[i]);
            TriMesh::VertexVertexIter vv_it = m_mesh.vv_iter(TriMesh::VertexHandle(ring[i]));
            for(;vv_it.is_valid();vv_it++)
            {
                next_ring.push_back((*vv_it).idx());
            }
        }
        ring = next_ring;
    }
    mean_normal*=0.0;
    for(std::set<int>::iterator iter=neighbors.begin(); iter!= neighbors.end(); iter++)
    {
        mean_normal+=m_mesh.normal(TriMesh::VertexHandle(*iter));
    }
    mean_normal/=neighbors.size();
    mean_normal/=sqrt(mean_normal|mean_normal);
}

void whole_3dmm_face::get_neighborIds_around_v(int v_id, std::vector<int> &ids, int neighbor_size)
{
    std::set<int> neighbors;
    std::vector<int> ring;
    ring.push_back(v_id);
    for(int time=0;time<neighbor_size;time++)
    {
        std::vector<int> next_ring;
        for(int i=0;i<ring.size();i++)
        {
            if(neighbors.count(ring[i]))
                continue;
            neighbors.insert(ring[i]);
            TriMesh::VertexVertexIter vv_it = m_mesh.vv_iter(TriMesh::VertexHandle(ring[i]));
            for(;vv_it.is_valid();vv_it++)
            {
                next_ring.push_back((*vv_it).idx());
            }
        }
        ring = next_ring;
    }
    ids.clear();
    for(std::set<int>::iterator iter=neighbors.begin(); iter!= neighbors.end(); iter++)
    {
        ids.push_back(*iter);
    }
}

int whole_3dmm_face::get_shape_pcanum()
{
    return m_shape_pcanum;
}

int whole_3dmm_face::get_exp_pcanum()
{
    return m_exp_pcanum;
}

int whole_3dmm_face::get_keypoints_size()
{
    return m_whole_keypoints.size();
}

const std::vector<int> &whole_3dmm_face::get_keypoints()
{
    return m_whole_keypoints;
}

int whole_3dmm_face::get_keypoint_id(int i)
{
    CHECK_GE(i,0)<<"whole_3dmm_face::get_keypoint_id id "<<i<<" range wrong!";
    CHECK_LT(i,get_keypoints_size())<<"whole_3dmm_face::get_keypoint_id id "<<i<<" range wrong!";

    return m_whole_keypoints[i];
}

const MatrixXf &whole_3dmm_face::get_shape_st()
{
    return m_shape_st;
}

const MatrixXf &whole_3dmm_face::get_exp_st()
{
    return m_exp_st;
}

void whole_3dmm_face::get_mean_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    Eigen::MatrixXf meantemp=m_mean_exp+m_mean_shape;
    meantemp.resize(meantemp.size(),1);
    base.resize(ids.size()*3,1);
    for(int i=0;i<ids.size();i++)
    {
        base.block(3*i,0,3,1) = meantemp.block(3*ids[i],0,3,1);
    }
}

void whole_3dmm_face::get_shape_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    base.resize(ids.size()*3,m_shape_pcanum);
    for(int i=0;i<ids.size();i++)
    {
        base.block(3*i,0,3,m_shape_pcanum) = m_pca_shape.block(3*ids[i],0,3,m_shape_pcanum);
    }
}

void whole_3dmm_face::get_exp_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    base.resize(ids.size()*3,m_exp_pcanum);
    for(int i=0;i<ids.size();i++)
    {
        base.block(3*i,0,3,m_exp_pcanum) = m_pca_exp.block(3*ids[i],0,3,m_exp_pcanum);
    }
}

void whole_3dmm_face::get_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    base.resize(3*ids.size(),m_shape_pcanum+m_exp_pcanum);
    Eigen::MatrixXf temp;
    get_shape_vertex_base_on_ids(ids,temp);
    base.block(0,0,3*ids.size(),m_shape_pcanum)=temp;
    get_exp_vertex_base_on_ids(ids,temp);
    base.block(0,m_shape_pcanum,3*ids.size(),m_exp_pcanum)=temp;
}

void whole_3dmm_face::initial()
{
    if(!is_3DMM_initialized)
    {
        io_utils::read_pca_models("../resource/zhu_3dmm/ShapePCA.bin",m_mean_shape,m_pca_shape,m_shape_st,
                                  m_whole_vnum,m_shape_pcanum);
        io_utils::read_pca_models("../resource/zhu_3dmm/ExpPCA.bin",m_mean_exp,m_pca_exp,m_exp_st,
                                  m_whole_vnum,m_exp_pcanum);
        io_utils::read_all_type_file_to_vector<int>("../resource/zhu_3dmm/whole_MultiPie68_keypoints.txt",m_whole_keypoints);
        is_3DMM_initialized = true;
    }

    OpenMesh::IO::read_mesh(m_mesh,"../resource/zhu_3dmm/mesh.obj");
    m_mesh.request_vertex_normals();
    m_mesh.request_face_normals();


    m_shape.resize(m_shape_pcanum);
    m_exp.resize(m_exp_pcanum);

    is_updated=true;
}

void whole_3dmm_face::update_mesh()
{
    VectorXf verts = m_mean_exp+m_mean_shape+m_pca_shape*m_shape+m_pca_exp*m_exp;
    TriMesh::VertexIter v_it = m_mesh.vertices_begin();
    float *addrV = const_cast<float*>(m_mesh.point(*v_it).data());
    memcpy(addrV,verts.data(),sizeof(float)*verts.size());
    m_mesh.update_normals();
    is_updated=true;
}
