#include "draw_verify_result.h"
#include "io_utils.h"
#include <QDir>
#include <QStringList>
#include "glog/logging.h"
#include <opencv2/highgui/highgui.hpp>
draw_verify_result::draw_verify_result(const std::string &data_root, const std::string &result_root, const std::string &saveimg_root, int threads_num)
    :m_data_root(data_root),m_result_root(result_root),m_save_root(saveimg_root),m_threads_num(threads_num)
{
    m_3dmm_meshs.resize(m_threads_num,Face());
}

void draw_verify_result::draw(bool is_draw_dense, bool is_draw_unvisible)
{
    std::vector<int> ids;
    if(is_draw_dense)
        ids=Face::get_dense_keypoint();
    else
        ids=Face::get_part_keypoints();

    std::vector<std::string> meshfiles;
    std::vector<std::vector<std::string> > per_mesh_pose_files;
    read_result_file_names(meshfiles,per_mesh_pose_files);

    Eigen::VectorXi nums(m_threads_num);
    nums.setZero();
    LOG(INFO)<<"start draw result imgs..";
    #pragma omp parallel for num_threads(m_threads_num)
    for(int i=0;i<meshfiles.size();i++)
    {
        int thread_id=omp_get_thread_num();
        std::vector<float> temp;
        io_utils::read_all_type_file_to_vector<float>(m_result_root+meshfiles[i],temp);
        Eigen::VectorXf shape(Face::get_shape_pcanum());
        Eigen::VectorXf exp(Face::get_exp_pcanum());
        memcpy(shape.data(),temp.data(),sizeof(float)*shape.size());
        memcpy(exp.data(),temp.data()+Face::get_shape_pcanum(),sizeof(float)*exp.size());
        m_3dmm_meshs[thread_id].set_shape(shape);
        m_3dmm_meshs[thread_id].set_exp(exp);
        Eigen::MatrixXf verts;
        m_3dmm_meshs[thread_id].get_vertex_matrix(verts,false);
        std::string savemesh = m_save_root+meshfiles[i].substr(0,meshfiles[i].size()-14)+".obj";
        OpenMesh::IO::write_mesh(m_3dmm_meshs[thread_id].get_part_face(),savemesh);
        const std::vector<std::string> &posefiles = per_mesh_pose_files[i];
        for(int j=0;j<posefiles.size();j++)
        {
            io_utils::read_all_type_file_to_vector<float>(m_result_root+posefiles[j],temp);
            Eigen::VectorXf para(temp.size());
            memcpy(para.data(),temp.data(),sizeof(float)*para.size());
            float scale = para(0);
            float tx = para[4]; float ty = para[5];
            Eigen::Affine3f transformation;
            transformation  = Eigen::AngleAxisf(para(1), Eigen::Vector3f::UnitX()) *
                              Eigen::AngleAxisf(para(2), Eigen::Vector3f::UnitY()) *
                              Eigen::AngleAxisf(para(3), Eigen::Vector3f::UnitZ());
            Eigen::Matrix3f R = transformation.rotation();
            std::vector<bool> visuals;
            compute_keypoint_visible_multi_thread(R,ids,visuals,thread_id);

            Eigen::MatrixXf temp_v(3,ids.size());
            for(int k=0;k<ids.size();k++)
                temp_v.col(k) = verts.col(ids[k]);
            temp_v = R*temp_v;
            temp_v*=scale;
            Eigen::Vector3f trans;  trans(0) = tx;  trans(1) = ty;  trans(2) = 0.0;
            temp_v.colwise() += trans;
            Eigen::MatrixXf keypos = temp_v.block(0,0,2,ids.size());
            //convert to left up origin
            for(int k=0;k<ids.size();k++)
                keypos(1,k) = 224.0-keypos(1,k);
            std::string imgfile = posefiles[j].substr(0,posefiles[j].size()-9)+".jpg";
            cv::Mat img=cv::imread(m_data_root+imgfile,cv::IMREAD_COLOR);
            for(int k=0;k<ids.size();k++)
            {
                if(visuals[k])
                    cv::circle(img,cv::Point(keypos(0,k),keypos(1,k)),1,cv::Scalar(0,255,0),-1);
                if(!visuals[k]&&is_draw_unvisible)
                    cv::circle(img,cv::Point(keypos(0,k),keypos(1,k)),1,cv::Scalar(0,0,255),-1);
            }
            std::string savefile=imgfile;
            if(is_draw_dense)
                savefile = savefile.substr(0,savefile.size()-4)+"_densep.jpg";
            else
                savefile = savefile.substr(0,savefile.size()-4)+"_keyp.jpg";
            cv::imwrite(m_save_root+savefile,img);
        }
        nums[thread_id]=nums[thread_id]+1;
        LOG_IF(INFO,nums.sum()%100==0)<<"keypoints img draw have been computed 100 faces";
    }
    LOG(INFO)<<"done.";
}

void draw_verify_result::read_result_file_names(std::vector<std::string> &meshfiles, std::vector<std::vector<std::string> > &per_mesh_pose_files)
{
    LOG(INFO)<<"start read file names";
    QDir path(QString(m_result_root.data()));
    path.setFilter(QDir::Files);
    QStringList filters;
    filters.push_back("*_mesh_para.txt");
    path.setNameFilters(filters);
    path.setSorting(QDir::Name);
    QStringList entrys = path.entryList();
    meshfiles.clear();
    per_mesh_pose_files.clear();
    for(QStringList::Iterator vit = entrys.begin(); vit!=entrys.end(); vit++)
    {
        std::string meshfile = (*vit).toStdString();
        meshfiles.push_back(meshfile);
        QDir img_path(QString(m_result_root.data()));
        img_path.setFilter(QDir::Files);
        QStringList tfilters;
        tfilters.push_back(QString((meshfile.substr(0,meshfile.size()-14)+"_*_pose.txt").data()));
        img_path.setNameFilters(tfilters);
        img_path.setSorting(QDir::Name);
        QStringList imgentrys = img_path.entryList();
        std::vector<std::string> temp;
        for(QStringList::Iterator iit = imgentrys.begin(); iit!=imgentrys.end(); iit++)
        {
            temp.push_back((*iit).toStdString());
        }
        per_mesh_pose_files.push_back(temp);
    }
    LOG(INFO)<<"done";
}

void draw_verify_result::compute_keypoint_visible_multi_thread(const MatrixXf &R, const std::vector<int> &ids, std::vector<bool> &visuals, int thread_id)
{
    //this mesh's coordinate have not times Rotation
    visuals.resize(ids.size(), false);
//    TriMesh::Normal zdir(0.0,0.0,1.0);
    for(int i=0; i< ids.size(); i++)
    {
        int id = ids[i];
        TriMesh::Normal normal;
        m_3dmm_meshs[thread_id].get_mean_normal(id,normal);
        Eigen::Vector3f Rnormal = Eigen::Map<Eigen::Vector3f>(normal.data());
        Rnormal =R*Rnormal;
        float val = Rnormal(2);
        if(val>0.0)
        {
            visuals[i] = true;
        }
    }
}
