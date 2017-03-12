#include "face_img.h"
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include "glog/logging.h"
#include "io_utils.h"
face_img::face_img(int paranum, int keyposnum, const std::string &name)
{
    m_face_name = name;
    m_img_length = 224;
    m_para_num = paranum;
    m_keypos_num = keyposnum;
    m_groundtruth_land.resize(2*m_keypos_num);
    m_groundtruth_para.resize(m_para_num);
}

void face_img::read_img(const std::string &imgfile)
{
    cv::Mat tmp=cv::imread(imgfile,cv::IMREAD_GRAYSCALE);
    tmp.convertTo(tmp,CV_32F,1.0);
    if(tmp.rows!=m_img_length||tmp.cols!=m_img_length)
    {
        LOG(FATAL)<<"face_imgs::read_imgs read img"<<imgfile<<" size is wrong!";
        return;
    }
    m_img_data.resize(m_img_length*m_img_length,0.0);
    float *now_data = m_img_data.data();
    for(int i=0; i<m_img_length; i++)
    {
        //left top origin
        const float *ptr = tmp.ptr<float>(i);
        memcpy(now_data, ptr, sizeof(float)*m_img_length);
        now_data+=m_img_length;
    }
}

void face_img::read_para(const std::string &posefile)
{
    std::vector<float> para;
    io_utils::read_all_type_file_to_vector<float>(posefile, para);
    LOG_IF(FATAL,para.size()!=m_para_num)<<posefile<<" read pose para number is wrong!";
    m_groundtruth_para.resize(6);
    m_groundtruth_para(0)=para[5];
    m_groundtruth_para(1)=-para[0];  m_groundtruth_para(2)=-para[1];  m_groundtruth_para(3)=-para[2];
    m_groundtruth_para(4)=para[3];
    m_groundtruth_para(5)=para[4];
}

void face_img::read_land(const std::string &landfile)
{
    std::vector<float> lands;
    io_utils::read_all_type_file_to_vector<float>(landfile,lands);
    LOG_IF(FATAL,lands.size()!=2*m_keypos_num)<<landfile<<" read landmarks number is wrong!";
    m_groundtruth_land.resize(2*m_keypos_num);
    memcpy(m_groundtruth_land.data(),lands.data(),sizeof(float)*lands.size());
}
