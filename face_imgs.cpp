#include "face_imgs.h"
#include <QString>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "io_utils.h"
face_imgs::face_imgs(const std::string &root, const std::string &meshparas_name,int shape_dim, int exp_dim)
{
    QString temp(meshparas_name.data());
    temp.remove(temp.size()-14,14);
    m_files_root=root;
    m_face_name = temp.toStdString();
    m_shape_dimension=shape_dim;
    m_exp_dimension=exp_dim;
    initial();
}

void face_imgs::set_img_num(const std::vector<std::string>& imgfiles_name)
{
    m_imgfiles_name = imgfiles_name;
	m_img_num = m_imgfiles_name.size();
	resize_data();
}

void face_imgs::read_imgs()
{
    for(int i=0;i<m_img_num;i++)
    {
        QString num;
        num.setNum(i);
        std::string imgname=m_imgfiles_name[i];
        cv::Mat tmp=cv::imread(m_files_root+imgname,cv::IMREAD_GRAYSCALE);
        tmp.convertTo(tmp,CV_32F,1.0);
        if(tmp.rows!=m_img_length||tmp.cols!=m_img_length)
        {
			std::cout<<"face_imgs::read_imgs read img"<<imgname<<" size is wrong!"<<std::endl;			
            exit(1);
        }
        std::vector<float> img;
        img.resize(m_img_length*m_img_length,0.0);
        float *now_data = img.data();
        for(int i=0; i<m_img_length; i++)
        {
//            //left down origin
//            const float *ptr = tmp.ptr<float>(m_img_length-i-1);
            //left top origin
            const float *ptr = tmp.ptr<float>(i);
            memcpy(now_data, ptr, sizeof(float)*m_img_length);
            now_data+=m_img_length;
        }
        m_imgs_data[i]=img;
    }
}

void face_imgs::read_pose_paras()
{
    for(int i=0;i<m_img_num;i++)
    {
        QString num;
        num.setNum(i);
        QString tempname(m_imgfiles_name[i].data());
        tempname.remove(tempname.size()-4,4);
        std::string paraname=tempname.toStdString()+"_pose.txt";
        std::vector<float> paras;
        io_utils::read_all_type_file_to_vector<float>(m_files_root+paraname, paras);
//        Eigen::Affine3f transformation;
//        transformation  = Eigen::AngleAxisf(-paras[0], Eigen::Vector3f::UnitX()) *
//                          Eigen::AngleAxisf(-paras[1], Eigen::Vector3f::UnitY()) *
//                          Eigen::AngleAxisf(-paras[2], Eigen::Vector3f::UnitZ());
//        R = transformation.rotation();
//        weak_T(0)=paras[3];
//        weak_T(1)=paras[4];
//        scale=paras[5];
        Eigen::VectorXf temp(6);
        temp(0)=paras[5];
        temp(1)=-paras[0];  temp(2)=-paras[1];  temp(3)=-paras[2];
        temp(4)=paras[3];
        temp(5)=paras[4];
        m_groundtruth_paras.col(i) = temp;
    }
}

void face_imgs::read_shape_exp()
{
    std::vector<std::vector<float> > paras;
    io_utils::read_all_type_rowsfile_to_2vector<float>(m_files_root+m_face_name+"_mesh_para.txt", paras);
    if(m_shape_dimension>paras[0].size()||m_exp_dimension>paras[1].size())
    {
        std::cout<<"face_imgs::read_shape_exp vector dimention is wrong!"<<std::endl;
        exit(1);
    }
    memcpy(m_groundtruth_shapes.data(),paras[0].data(),m_shape_dimension*sizeof(float));
    memcpy(m_groundtruth_exps.data(),paras[1].data(),m_exp_dimension*sizeof(float));
}

void face_imgs::initial()
{
    m_img_length=224;
    m_img_num=0;
    m_groundtruth_shapes.resize(m_shape_dimension);
    m_groundtruth_exps.resize(m_exp_dimension);
}

void face_imgs::resize_data()
{
    m_imgs_data.resize(m_img_num,std::vector<float>(m_img_length*m_img_length,0.0));
    m_groundtruth_paras.resize(6,m_img_num);
}
