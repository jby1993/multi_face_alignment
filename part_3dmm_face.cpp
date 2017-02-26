#include "part_3dmm_face.h"
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
static std::vector<int> m_part_keypoints;
static std::vector<int> m_part_dense_keypoints;
static std::vector<int> m_partv_2_wholev;
static bool is_3DMM_initialized=false;

part_3DMM_face::part_3DMM_face()
{
    initial();
}

void part_3DMM_face::set_shape(const VectorXf &shape)
{
    m_shape=shape;
    is_whole_updated=false;
    is_part_updated=false;
}

void part_3DMM_face::set_exp(const VectorXf &exp)
{
    m_exp = exp;
    is_whole_updated=false;
    is_part_updated=false;
}

void part_3DMM_face::update_mesh(bool is_whole)
{
    VectorXf verts = m_mean_exp+m_mean_shape+m_pca_shape*m_shape+m_pca_exp*m_exp;
    if(!is_whole)
    {
        for(TriMesh::VertexIter vit=m_part_face.vertices_begin(); vit!=m_part_face.vertices_end(); vit++)
        {
            int pid = (*vit).idx();
            int wid = m_partv_2_wholev[pid];
            m_part_face.set_point(*vit, TriMesh::Point(verts(3*wid), verts(3*wid+1), verts(3*wid+2)));
        }
        m_part_face.update_normals();
        is_part_updated=true;
    }
    else
    {
        TriMesh::VertexIter v_it = m_whole_face.vertices_begin();
        float *addrV = const_cast<float*>(m_whole_face.point(*v_it).data());
        memcpy(addrV,verts.data(),sizeof(float)*verts.size());
        m_whole_face.update_normals();
        is_whole_updated=true;
    }
}

void part_3DMM_face::get_vertex_matrix(MatrixXf &verts, bool is_whole)
{
    if(is_whole)
    {
        if(!is_whole_updated)
            update_mesh(true);
        verts.resize(3,m_whole_face.n_vertices());
        TriMesh::VertexIter v_it = m_whole_face.vertices_begin();
        const float *addrV = m_whole_face.point(*v_it).data();
        memcpy(verts.data(),addrV,sizeof(float)*verts.size());
    }
    else
    {
        if(!is_part_updated)
            update_mesh(false);
        verts.resize(3,m_part_face.n_vertices());
        TriMesh::VertexIter v_it = m_part_face.vertices_begin();
        const float *addrV = m_part_face.point(*v_it).data();
        memcpy(verts.data(),addrV,sizeof(float)*verts.size());
    }
}

const TriMesh &part_3DMM_face::get_part_face()
{
    if(!is_part_updated)
        update_mesh(false);
    return m_part_face;
}

void part_3DMM_face::get_mean_normal(int v_id, TriMesh::Normal &mean_normal,int neighbor_size, bool is_whole)
{
    if(!is_whole_updated)   update_mesh(true);
    if(!is_part_updated)    update_mesh(false);
    TriMesh *mesh;
    if(is_whole)    mesh = &m_whole_face;
    else    mesh = &m_part_face;
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
            TriMesh::VertexVertexIter vv_it = mesh->vv_iter(TriMesh::VertexHandle(ring[i]));
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
        mean_normal+=mesh->normal(TriMesh::VertexHandle(*iter));
    }
    mean_normal/=neighbors.size();
    mean_normal/=sqrt(mean_normal|mean_normal);
}

int part_3DMM_face::get_shape_pcanum()
{
    return m_shape_pcanum;
}

int part_3DMM_face::get_exp_pcanum()
{
    return m_exp_pcanum;
}

int part_3DMM_face::get_keypoints_size()
{
    return m_part_keypoints.size();
}

int part_3DMM_face::get_dense_keypoint_size()
{
    return m_part_dense_keypoints.size();
}

const std::vector<int> &part_3DMM_face::get_part_keypoints()
{
    return m_part_keypoints;
}

int part_3DMM_face::get_dense_keypoint_id(int i)
{
    CHECK_GE(i,0)<<"part_3DMM_face::get_dense_keypoint_id id range wrong!";
    CHECK_LT(i,get_dense_keypoint_size())<<"part_3DMM_face::get_dense_keypoint_id id range wrong!";
    return m_part_dense_keypoints[i];
}

const std::vector<int> &part_3DMM_face::get_dense_keypoint()
{
    return m_part_dense_keypoints;
}

const std::vector<int> &part_3DMM_face::get_partv2wholev()
{
    return m_partv_2_wholev;
}

const MatrixXf &part_3DMM_face::get_shape_st()
{
    return m_shape_st;
}

const MatrixXf &part_3DMM_face::get_exp_st()
{
    return m_exp_st;
}

void part_3DMM_face::get_mean_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    std::vector<int> wids;
    int partvnum=m_partv_2_wholev.size();
    for(int i=0;i<ids.size();i++)
    {
        LOG_IF(FATAL,ids[i]>=partvnum)<<"ids input is illegal!";
        wids.push_back(m_partv_2_wholev[ids[i]]);
    }
    Eigen::MatrixXf meantemp=m_mean_exp+m_mean_shape;
    meantemp.resize(meantemp.size(),1);
    base.resize(wids.size()*3,1);
    for(int i=0;i<wids.size();i++)
    {
        base.block(3*i,0,3,1) = meantemp.block(3*wids[i],0,3,1);
    }
}

void part_3DMM_face::get_shape_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    std::vector<int> wids;
    int partvnum=m_partv_2_wholev.size();
    for(int i=0;i<ids.size();i++)
    {
        LOG_IF(FATAL,ids[i]>=partvnum)<<"ids input is illegal!";
        wids.push_back(m_partv_2_wholev[ids[i]]);
    }
    base.resize(wids.size()*3,m_shape_pcanum);
    for(int i=0;i<wids.size();i++)
    {
        base.block(3*i,0,3,m_shape_pcanum) = m_pca_shape.block(3*wids[i],0,3,m_shape_pcanum);
    }
}

void part_3DMM_face::get_exp_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    std::vector<int> wids;
    int partvnum=m_partv_2_wholev.size();
    for(int i=0;i<ids.size();i++)
    {
        LOG_IF(FATAL,ids[i]>=partvnum)<<"ids input is illegal!";
        wids.push_back(m_partv_2_wholev[ids[i]]);
    }
    base.resize(wids.size()*3,m_exp_pcanum);
    for(int i=0;i<wids.size();i++)
    {
        base.block(3*i,0,3,m_exp_pcanum) = m_pca_exp.block(3*wids[i],0,3,m_exp_pcanum);
    }
}

void part_3DMM_face::get_vertex_base_on_ids(const std::vector<int> &ids, MatrixXf &base)
{
    base.resize(3*ids.size(),m_shape_pcanum+m_exp_pcanum);
    Eigen::MatrixXf temp;
    get_shape_vertex_base_on_ids(ids,temp);
    base.block(0,0,3*ids.size(),m_shape_pcanum)=temp;
    get_exp_vertex_base_on_ids(ids,temp);
    base.block(0,m_shape_pcanum,3*ids.size(),m_exp_pcanum)=temp;
}

void part_3DMM_face::initial()
{
    if(!is_3DMM_initialized)
    {
        io_utils::read_pca_models("../resource/mainShapePCA.bin",m_mean_shape,m_pca_shape,m_shape_st,
                                  m_whole_vnum,m_shape_pcanum);
        io_utils::read_pca_models("../resource/DeltaExpPCA.bin",m_mean_exp,m_pca_exp,m_exp_st,
                                  m_whole_vnum,m_exp_pcanum);
        io_utils::read_all_type_file_to_vector<int>("../resource/partv23dmmv.txt",m_partv_2_wholev);
        io_utils::read_all_type_file_to_vector<int>("../resource/part_face_keypoints.txt",m_part_keypoints);
        io_utils::read_all_type_file_to_vector<int>("../resource/part_dense_keypoints.txt",m_part_dense_keypoints);
        m_whole_keypoints.clear();
        for(int i=0;i<m_part_keypoints.size();i++)
            m_whole_keypoints.push_back(m_partv_2_wholev[m_part_keypoints[i]]);
        is_3DMM_initialized = true;
    }

    OpenMesh::IO::read_mesh(m_part_face,"../resource/part_face.obj");
    m_part_face.request_vertex_normals();
    m_part_face.request_face_normals();

    OpenMesh::IO::read_mesh(m_whole_face,"../resource/whole_face.obj");
    m_whole_face.request_vertex_normals();
    m_whole_face.request_face_normals();

    m_shape.resize(m_shape_pcanum);
    m_exp.resize(m_exp_pcanum);

    is_whole_updated=true;
    is_part_updated=true;
}
