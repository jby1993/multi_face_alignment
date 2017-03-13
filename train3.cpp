#include "train3.h"
#include "io_utils.h"
#include "train.h"
#include <QString>
train3::train3(int thread_num)
{
    m_img_root = "../../multi_learn_data/train_300_LP/clip_imgs/";
    m_savemodel_root = "../save_model/";
    m_para_mean_sd_file="../resource/300WLP_para_mean_sd.bin";
    m_para_num = 6;
    m_casscade_sum = 6;
    m_threadnum_for_compute_features=thread_num;
    for(int i=0;i<m_threadnum_for_compute_features;i++)
    {
//        m_feature_detectors.push_back(CNNDenseFeature());
        m_3dmm_meshs.push_back(part_3DMM_face());
//        std::cout<<i<<std::endl;
    }
//    m_feature_detectors.resize(m_threadnum_for_compute_features,CNNDenseFeature());
//    m_3dmm_meshs.resize(m_threadnum_for_compute_features,part_3DMM_face());
    m_gpuid_for_feature_computes.resize(1,0);
    m_gpuid_for_matrix_compute=0;
#ifdef USE_CNNFEATURE
    m_feature_size=64;
#else
    m_feature_size=128;
#endif
    compute_shapes_exps_mean_sd();
    read_para_mean_sd();
}

void train3::read_img_datas(const std::string &imglist_file)
{
    std::vector<std::string> imglist;
    io_utils::read_all_type_file_to_vector<std::string>(imglist_file,imglist);
    LOG(INFO)<<"start read imgs...";
    for(int i=0; i<imglist.size(); i++)
    {
        std::string imgname = imglist[i];
        std::string posename = imgname;
        QString tstr(posename.data());
        tstr.remove(tstr.size()-4,4);
        std::string facename = tstr.toStdString();
        posename = facename;
        posename += "_pose.txt";
        std::string landname = imgname;
        tstr.clear();
        tstr.append(landname.data());
        tstr.remove(tstr.size()-4,4);
        landname = tstr.toStdString();
        landname += "_keypos.txt";
        face_img temp(m_para_num,Face::get_whole_keypoint_size(),facename);
        temp.read_img(m_img_root+imgname);
        temp.read_para(m_poses_root+posename);
        temp.read_land(m_lands_root+landname);
        m_face_imgs.push_back(temp);
        LOG_IF(INFO,i%1000==999)<<i<<" imgs have been readed.";
    }
    m_img_num = m_face_imgs.size();
    LOG(INFO)<<"read "<<m_img_num<<" imgs";

    //initial need by multithread feature computation variables
    m_face_imgs_pointer.clear();
    std::list<face_img>::iterator iter=m_face_imgs.begin();
    for(;iter!=m_face_imgs.end();iter++)
        m_face_imgs_pointer.push_back(&(*iter));

    m_train_paras.resize(m_para_num,m_img_num);
    m_train_shapes.resize(Face::get_shape_pcanum(),m_img_num);
    m_train_exps.resize(Face::get_exp_pcanum(),m_img_num);
    m_visible_features.resize(Face::get_whole_keypoint_size()*m_feature_size,m_img_num);
    m_train_keypos.resize(2*Face::get_whole_keypoint_size(),m_img_num);
    m_keypos_visible.resize(Face::get_whole_keypoint_size(),m_img_num);

}

void train3::train_model()
{
    initial_shape_exp_with_mean();
    initial_para();
    compute_keypos_visibility();
    m_casscade_level=-1;
    show_delta_keypos_info();
    show_delta_para_info();
    m_keypos_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    m_para_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    int pre_trained_num = 3;
    for(int i=0;i<pre_trained_num;i++)
    {
        read_keypos_R(m_savemodel_root,i);
        read_para_R(m_savemodel_root,i);
    }

    for(m_casscade_level=0; m_casscade_level<m_casscade_sum; m_casscade_level++)
    {
        compute_all_visible_features_multi_thread();
        if(m_casscade_level<pre_trained_num)
            update_keypos_R();
        else
            compute_update_keypos_R();
        show_delta_keypos_info();
        if(m_casscade_level<pre_trained_num)
            update_para_R();
        else
            compute_update_para_R();
        show_delta_para_info();
        save_keypos_R(m_casscade_level);
        save_paras_R(m_casscade_level);

        optimize_shape_exp();
        compute_keypos_visibility();
    }
}

void train3::test_model()
{
    initial_shape_exp_with_mean();
    initial_para();
    compute_keypos_visibility();
    m_casscade_level=-1;
    for(m_casscade_level=0; m_casscade_level<m_casscade_sum-1; m_casscade_level++)
    {
        compute_all_visible_features_multi_thread();
        update_keypos_R();
        update_para_R();

        optimize_shape_exp();
        compute_keypos_visibility();
    }
    compute_all_visible_features_multi_thread();
    update_keypos_R();
    update_para_R();
    optimize_shape_exp();
}

void train3::save_verify_result(const std::string &root)
{
    for(int i=0;i<m_face_imgs_pointer.size();i++)
    {
        face_img *data = m_face_imgs_pointer[i];
        std::string file=root+data->get_face_name()+"_mesh_para.txt";
        Eigen::VectorXf temp(Face::get_shape_pcanum()+Face::get_exp_pcanum());
        temp.block(0,0,Face::get_shape_pcanum(),1) = m_train_shapes.col(i);
        temp.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),1) = m_train_exps.col(i);
        io_utils::write_all_type_to_file<float>(temp,file);

        file = root+data->get_face_name()+"_pose.txt";
        io_utils::write_all_type_to_file<float>(m_train_paras.col(i),file);

        file = root+data->get_face_name()+"_keypos.txt";
        io_utils::write_all_type_to_file<float>(m_train_keypos.col(i),file);

        file = root+data->get_face_name()+"_visi.txt";
        io_utils::write_all_type_to_file<int>(m_keypos_visible.col(i),file);

    }
}

void train3::read_trained_model(const std::string &root, int casscade_num)
{
    m_casscade_sum = casscade_num;
    m_keypos_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    m_para_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    for(int i=0;i<casscade_num;i++)
    {
        read_keypos_R(root,i);
        read_para_R(root,i);
    }
}

void train3::set_feature_compute_gpu(const std::vector<int> ids)
{
    m_feature_detectors.clear();
    bool normalized = false;
    if(ids.size()!=m_threadnum_for_compute_features)
    {
        LOG(FATAL)<<"train::set_feature_compute_gpu ids size wrong";
        m_gpuid_for_feature_computes.resize(1,0);
#ifdef USE_CNNFEATURE
        m_feature_detectors.push_back(CNNDenseFeature(0,normalized));
#else
       m_feature_detectors.push_back(SIFTDectector());
       LOG(INFO)<<"SIFT dectector 0 has been build.";
#endif
    }
    m_gpuid_for_feature_computes=ids;
    for(int i=0;i<m_threadnum_for_compute_features;i++)
    {
#ifdef USE_CNNFEATURE
        m_feature_detectors.push_back(CNNDenseFeature(m_gpuid_for_feature_computes[i],normalized));
        LOG(INFO)<<"Net "<<i<<" has been build.";
#else
       m_feature_detectors.push_back(SIFTDectector());
       LOG(INFO)<<"SIFT dectector "<<i<<" has been build.";
#endif
    }
}

void train3::initial_shape_exp_with_mean()
{
    m_train_shapes.setZero();
    m_train_exps.setZero();
}

void train3::initial_para()
{
    for(int i=0; i<m_img_num; i++)
        m_train_paras.col(i) = m_groundtruth_paras_mean;
}

void train3::compute_keypos_visibility()
{
    LOG(INFO)<<"start compute keypos and visibility...";
    Eigen::VectorXi nums(m_threadnum_for_compute_features);
    nums.setZero();
    #pragma omp parallel for num_threads(m_threadnum_for_compute_features)
    for(int i=0;i<m_face_imgs_pointer.size();i++)
    {
        int thread_id=omp_get_thread_num();
        face_img *data = m_face_imgs_pointer[i];
        m_3dmm_meshs[thread_id].set_shape(m_train_shapes.col(i));
        m_3dmm_meshs[thread_id].set_exp(m_train_exps.col(i));
        MatrixXf verts;
        m_3dmm_meshs[thread_id].get_vertex_matrix(verts, true);

        float *para = m_train_paras.col(i).data();
        float scale = para[0];
        float ax = para[1]; float ay = para[2]; float az = para[3];
        float tx = para[4]; float ty = para[5];

        Eigen::Affine3f transformation;
        transformation  = Eigen::AngleAxisf(ax, Eigen::Vector3f::UnitX()) *
                          Eigen::AngleAxisf(ay, Eigen::Vector3f::UnitY()) *
                          Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f R = transformation.rotation();
        std::vector<bool> visuals;
        compute_keypoint_visible_multi_thread(R,Face::get_whole_keypoint(),visuals,thread_id);
        Eigen::MatrixXf temp_v(3,Face::get_whole_keypoint_size());
        for(int k=0;k<Face::get_whole_keypoint_size();k++)
            temp_v.col(k) = verts.col(Face::get_whole_keypoint_id(k));
        temp_v = R*temp_v;
        temp_v*=scale;
        Eigen::Vector3f trans;  trans(0) = tx;  trans(1) = ty;  trans(2) = 0.0;
        temp_v.colwise() += trans;
        Eigen::MatrixXf temp = temp_v.block(0,0,2,Face::get_whole_keypoint_size());
        //convert to left up origin
        for(int k=0;k<Face::get_whole_keypoint_size();k++)
            temp(1,k) = data->img_length()-temp(1,k);
        temp.resize(temp.size(),1);
        m_train_keypos.col(i) = temp;
        for(int k=0;k<visuals.size();k++)
            if(visuals[k])
                m_keypos_visible(k,i) = 1;
            else
                m_keypos_visible(k,i) = 0;

        nums[thread_id]=nums[thread_id]+1;
        LOG_IF(INFO,nums.sum()%500==0)<<"img keypos visibility have been computed "<<nums.sum();
    }
    LOG(INFO)<<"done.";
}

void train3::read_para_mean_sd()
{
    std::vector<float> vals;
    io_utils::read_all_type_from_bin<float>(m_para_mean_sd_file,2*m_para_num,vals);
    m_groundtruth_paras_sd.resize(m_para_num);
    m_groundtruth_paras_mean.resize(m_para_num);
    Eigen::VectorXf temp(m_para_num);
    memcpy(temp.data(),vals.data(),m_para_num*sizeof(float));
    //chengxu li de para sunxu yu wenjian li de bu yi yang, duqu shi gai cheng yi zhi
    m_groundtruth_paras_mean(0) = temp(5);
    m_groundtruth_paras_mean(1)=-temp[0];  m_groundtruth_paras_mean(2)=-temp[1];  m_groundtruth_paras_mean(3)=-temp[2];
    m_groundtruth_paras_mean(4)=temp[3];
    m_groundtruth_paras_mean(5)=temp[4];
    memcpy(temp.data(),vals.data()+m_para_num,m_para_num*sizeof(float));
    //chengxu li de para sunxu yu wenjian li de bu yi yang, duqu shi gai cheng yi zhi
    m_groundtruth_paras_sd(0) = temp(5);
    m_groundtruth_paras_sd(1)=-temp[0];  m_groundtruth_paras_sd(2)=-temp[1];  m_groundtruth_paras_sd(3)=-temp[2];
    m_groundtruth_paras_sd(4)=temp[3];
    m_groundtruth_paras_sd(5)=temp[4];
}

void train3::compute_shapes_exps_mean_sd()
{
    m_groundtruth_shapes_exps_sd.resize(Face::get_shape_pcanum()+Face::get_exp_pcanum());
    m_groundtruth_shapes_exps_sd.block(0,0,Face::get_shape_pcanum(),1)=Face::get_shape_st();
    m_groundtruth_shapes_exps_sd.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),1)=Face::get_exp_st();

}

void train3::compute_keypoint_visible_multi_thread(const MatrixXf &R, const std::vector<int> &ids, std::vector<bool> &visuals, int thread_id)
{
    //this mesh's coordinate have not times Rotation
    visuals.resize(ids.size(), false);
//    TriMesh::Normal zdir(0.0,0.0,1.0);
    for(int i=0; i< ids.size(); i++)
    {
        int id = ids[i];
        TriMesh::Normal normal;
        m_3dmm_meshs[thread_id].get_mean_normal(id,normal,3,true);
        Eigen::Vector3f Rnormal = Eigen::Map<Eigen::Vector3f>(normal.data());
        Rnormal =R*Rnormal;
        float val = Rnormal(2);
        if(val>0.0)
        {
            visuals[i] = true;
        }
    }
}

void train3::compute_delta_para(MatrixXf &delta_para)
{
    delta_para.resize(m_para_num,m_img_num);
    for(int i=0; i<m_face_imgs_pointer.size(); i++)
    {
            delta_para.col(i) = m_face_imgs_pointer[i]->get_groundtruth_para()-m_train_paras.col(i);
    }
}

void train3::compute_delta_keypos(MatrixXf &delta_keypos)
{
    delta_keypos.resize(2*Face::get_whole_keypoint_size(),m_img_num);
    for(int i=0; i<m_face_imgs_pointer.size(); i++)
    {
            delta_keypos.col(i) = m_face_imgs_pointer[i]->get_groundtruth_land()-m_train_keypos.col(i);
    }
}

void train3::show_delta_para_info()
{
    Eigen::MatrixXf delta_para;
    compute_delta_para(delta_para);
    Eigen::MatrixXf delta_s = delta_para.row(0);
    Eigen::MatrixXf delta_r = delta_para.block(1,0,3,m_img_num);
    Eigen::MatrixXf delta_t = delta_para.block(4,0,2,m_img_num);
    Eigen::MatrixXf delta_norm = delta_s.colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" scale with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();
    delta_norm = (delta_r.row(0)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" ax with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();
    delta_norm = (delta_r.row(1)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" ay with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();
    delta_norm = (delta_r.row(2)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" az with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();

    delta_norm = delta_t.colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" t with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();
}

void train3::show_delta_keypos_info()
{
    Eigen::MatrixXf delta_pos;
    compute_delta_keypos(delta_pos);
    //take 3 pos to show delta info
    Eigen::MatrixXf temp = delta_pos.block(2*0,0,2,m_img_num);
    Eigen::MatrixXf delta_norm = temp.colwise().norm();
    LOG(INFO)<<"casscade keypos "<<m_casscade_level<<" keypose 0 with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();

    temp = delta_pos.block(2*30,0,2,m_img_num);
    delta_norm = temp.colwise().norm();
    LOG(INFO)<<"casscade keypos "<<m_casscade_level<<" keypose 30 with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();

    temp = delta_pos.block(2*60,0,2,m_img_num);
    delta_norm = temp.colwise().norm();
    LOG(INFO)<<"casscade keypos "<<m_casscade_level<<" keypose 60 with ground truth norm: mean: "<<delta_norm.mean()
            <<" max: "<<delta_norm.maxCoeff()<<" min: "<<delta_norm.minCoeff();
}

void train3::compute_all_visible_features_multi_thread()
{
    Eigen::VectorXi nums(m_threadnum_for_compute_features);
    nums.setZero();
    LOG(INFO)<<"start compute visible features";
    #pragma omp parallel for num_threads(m_threadnum_for_compute_features)
    for(int i=0;i<m_face_imgs_pointer.size();i++)
    {

        int thread_id=omp_get_thread_num();
#ifdef USE_CNNFEATURE
        //for Caffe set thread independent Caffe setting, this is essential, otherwise non-major thread caffe will
        //not take the setting on CNNDenseFeature, like setmode(GPU), still default CPU
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(m_gpuid_for_feature_computes[thread_id]);
#endif
        face_img *data = m_face_imgs_pointer[i];
        Eigen::VectorXf visible_features(m_feature_size*Face::get_whole_keypoint_size());
        Eigen::MatrixXf keypos;
        keypos=m_train_keypos.col(i);
        keypos.resize(2,keypos.size()/2);
        std::vector<bool> visibility(m_keypos_visible.rows(),false);
        for(int k=0;k<m_keypos_visible.rows();k++)
            if(m_keypos_visible(k,i)==1)
                visibility[k] = true;
            else
                visibility[k] = false;
#ifdef USE_CNNFEATURE
        m_feature_detectors[thread_id].set_data(data->get_img_data());
        m_feature_detectors[thread_id].get_compute_visible_posfeatures(keypos,visibility,visible_features);
#else
        Eigen::VectorXf scales(Face::get_whole_keypoint_size());
        scales.setOnes();
        scales *= 6.0;
        m_feature_detectors[thread_id].DescriptorOnCustomPoints(data->get_img(),data->img_length(),data->img_length(),
                                                                visibility,keypos,scales,visible_features);
#endif
        m_visible_features.col(i) = visible_features;

        nums[thread_id]=nums[thread_id]+1;
        LOG_IF(INFO,nums.sum()%500==0)<<"face img features have been computed "<<nums.sum();
    }
    LOG(INFO)<<"done!";
}

void train3::compute_update_keypos_R()
{
#ifdef USE_CNNFEATURE
    float lamda = 10.0;
#else
    float lamda = 5.0;
#endif
    MatrixXf f(m_visible_features.rows()+1,m_visible_features.cols());
    f.block(0,0,m_visible_features.rows(),m_visible_features.cols()) = m_visible_features;
    f.row(f.rows()-1).setOnes();
    MatrixXf delta_pos;
    compute_delta_keypos(delta_pos);
    MatrixXf &R=m_keypos_Rs[m_casscade_level];
    LOG(INFO)<<"casscade keypos "<<m_casscade_level<<" start computing...";
    compute_R(delta_pos,f,lamda,R);
    LOG(INFO)<<"casscade keypos "<<m_casscade_level<<" result norm: "<<R.norm()<<
        " delta mean norm: "<<(delta_pos-R*f).colwise().norm().mean();
    //update
    m_train_keypos = m_train_keypos+R*f;
}

void train3::compute_update_para_R()
{
#ifdef USE_CNNFEATURE
    float lamda = 500.0;
#else
    float lamda = 5.0;
#endif
    //use feature to infer delta_para
    MatrixXf f(m_visible_features.rows()+1,m_visible_features.cols());
    f.block(0,0,m_visible_features.rows(),m_visible_features.cols()) = m_visible_features;
    f.row(f.rows()-1).setOnes();
    MatrixXf delta_para;
    compute_delta_para(delta_para);
    //normalize paras
    delta_para = (1.0/m_groundtruth_paras_sd.array()).matrix().asDiagonal()*delta_para;
    MatrixXf &R = m_para_Rs[m_casscade_level];
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" start computing...";
    compute_R(delta_para,f,lamda,R);
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" result norm: "<<R.norm()<<
               " delta mean norm: "<<(delta_para-R*f).colwise().norm().mean();
    //update
    delta_para = R*f;
    //unnormalize paras
    delta_para = m_groundtruth_paras_sd.asDiagonal()*delta_para;
    m_train_paras = m_train_paras+delta_para;
}

void train3::save_keypos_R(int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name = m_savemodel_root+"keypos_Rs_"+num.toStdString()+".bin";
    save_R(m_keypos_Rs[casscade_level],name);
}

void train3::save_paras_R(int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name = m_savemodel_root+"paras_Rs_"+num.toStdString()+".bin";
    save_R(m_para_Rs[casscade_level],name);
}

void train3::optimize_shape_exp()
{
    LOG(INFO)<<"casscade shape exp "<<m_casscade_level<<" start optimizing...";
    Eigen::VectorXi nums(m_threadnum_for_compute_features);
    nums.setZero();
#ifdef USE_CNNFEATURE
    float lamda = 1.0;
#else
    float lamda = 0.01;
#endif
    Eigen::MatrixXf keypos_mean,keypos_base;
    Face::get_mean_vertex_base_on_ids(Face::get_whole_keypoint(),keypos_mean,true);
    Face::get_vertex_base_on_ids(Face::get_whole_keypoint(),keypos_base,true);
    #pragma omp parallel for num_threads(m_threadnum_for_compute_features)
    for(int i=0;i<m_face_imgs_pointer.size();i++)
    {
        int thread_id=omp_get_thread_num();
        face_img *data = m_face_imgs_pointer[i];
        Eigen::MatrixXf lhs(Face::get_whole_keypoint_size()*2+Face::get_shape_pcanum()+Face::get_exp_pcanum(),
                            Face::get_shape_pcanum()+Face::get_exp_pcanum());
        Eigen::MatrixXf rhs(Face::get_whole_keypoint_size()*2+Face::get_shape_pcanum()+Face::get_exp_pcanum(),1);

        float *para = m_train_paras.col(i).data();
        float scale = para[0];
        float ax = para[1]; float ay = para[2]; float az = para[3];
        float tx = para[4]; float ty = para[5];

        Eigen::Affine3f transformation;
        transformation  = Eigen::AngleAxisf(ax, Eigen::Vector3f::UnitX()) *
                          Eigen::AngleAxisf(ay, Eigen::Vector3f::UnitY()) *
                          Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f R = transformation.rotation();
        keypos_mean.resize(3,keypos_mean.size()/3);
        Eigen::MatrixXf temp = scale*R*keypos_mean;
        Eigen::Vector3f trans;  trans(0) = tx;  trans(1) = ty;  trans(2) = 0.0;
        temp = temp.colwise()+trans;
        Eigen::MatrixXf srmt = temp.block(0,0,2,temp.cols());
        Eigen::MatrixXf keypos = m_train_keypos.col(i);
        keypos.resize(2,keypos.size()/2);
        //convert origine to left bottom;
        for(int k=0;k<keypos.cols();k++)
            keypos(1,k) = data->img_length()-keypos(1,k);
        srmt.resize(srmt.size(),1);
        keypos.resize(keypos.size(),1);
        rhs.block(0,0,Face::get_whole_keypoint_size()*2,1)=keypos-srmt;

        temp.resize(keypos_base.rows(),keypos_base.cols());
        for(int k=0;k<Face::get_whole_keypoint_size();k++)
        {
            temp.block(3*k,0,3,temp.cols()) = scale*R*keypos_base.block(3*k,0,3,keypos_base.cols());
            lhs.block(2*k,0,2,lhs.cols()) = temp.block(3*k,0,2,temp.cols());
        }

        //add regular
        lhs.block(Face::get_whole_keypoint_size()*2,0,lhs.cols(),lhs.cols())=
                (lamda/m_groundtruth_shapes_exps_sd.array()).matrix().asDiagonal();
        rhs.block(Face::get_whole_keypoint_size()*2,0,lhs.cols(),1).setZero();
        //use jacobi decomposion to solve the least square problem
        Eigen::VectorXf x=lhs.jacobiSvd(ComputeThinU | ComputeThinV).solve(rhs);
        m_train_shapes.col(i) = x.block(0,0,Face::get_shape_pcanum(),1);
        m_train_exps.col(i) = x.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),1);

        nums[thread_id]=nums[thread_id]+1;
        LOG_IF(INFO,nums.sum()%500==0)<<"face img shape exp have been computed "<<nums.sum();
    }
    LOG(INFO)<<"done.";
}

void train3::update_keypos_R()
{
    MatrixXf f(m_visible_features.rows()+1,m_visible_features.cols());
    f.block(0,0,m_visible_features.rows(),m_visible_features.cols()) = m_visible_features;
    f.row(f.rows()-1).setOnes();
    MatrixXf &R=m_keypos_Rs[m_casscade_level];
    //update
    m_train_keypos = m_train_keypos+R*f;
}

void train3::update_para_R()
{
    MatrixXf f(m_visible_features.rows()+1,m_visible_features.cols());
    f.block(0,0,m_visible_features.rows(),m_visible_features.cols()) = m_visible_features;
    f.row(f.rows()-1).setOnes();
    MatrixXf &R = m_para_Rs[m_casscade_level];
    //update
    MatrixXf delta_para = R*f;
    //unnormalize paras
    delta_para = m_groundtruth_paras_sd.asDiagonal()*delta_para;
    m_train_paras = m_train_paras+delta_para;
}

void train3::compute_R(const MatrixXf &x, const MatrixXf &f, float lamda, MatrixXf &R)
{
    //    //using jacobi to solve
    //    Eigen::MatrixXf regu = (lamda*VectorXf(f.rows()).setOnes()).asDiagonal();
    //    MatrixXf lhs(f.cols()+f.rows(),f.rows());
    //    lhs.block(0,0,f.cols(),f.rows()) = f.transpose();
    //    lhs.block(f.cols(),0,f.rows(),f.rows()) = regu;
    //    MatrixXf rhs(x.cols()+f.rows(),x.rows());
    //    rhs.block(0,0,x.cols(),x.rows()) = x.transpose();
    //    rhs.block(x.cols(),0,f.rows(),x.rows()).setZero();
    //    LOG(INFO)<<"solve Rt with jacobi:"<<" compute for lhs("<<lhs.rows()<<"*"<<lhs.cols()<<")...";
    //    MatrixXf temp=lhs.jacobiSvd(ComputeThinU | ComputeThinV).solve(rhs);
    //    LOG(INFO)<<"done! ";
    //    LOG_IF(FATAL,check_matrix_invalid(temp))<<"computation become invalid! fatal wrong!";
    //    R = temp.transpose();

        //using ldlt to solve
        MatrixXf lhs(f.rows(),f.rows());
        lhs.setZero();
        int col_num = f.cols();
        int cols = 20000;
        int num = col_num/cols+1;
        //this is for save GPU memory
        for(int i=0;i<num-1;i++)
            train::my_gpu_rankUpdated(lhs,f.block(0,i*cols,f.rows(),cols),1.0,m_gpuid_for_matrix_compute);
        train::my_gpu_rankUpdated(lhs,f.block(0,(num-1)*cols,f.rows(),f.cols()-(num-1)*cols),1.0,m_gpuid_for_matrix_compute);
    //    lhs = f*f.transpose();
        lhs += (lamda*Eigen::VectorXf(f.rows()).setOnes()).asDiagonal();
        MatrixXf rhs(f.rows(),x.rows());
        rhs = f*x.transpose();
        lhs.triangularView<Eigen::Lower>() = lhs.transpose();
        LOG(INFO)<<"solve Rt with ldlt:"<<" compute for lhs("<<lhs.rows()<<"*"<<lhs.cols()<<")...";
        MatrixXf temp = lhs.ldlt().solve(rhs);
    //    MatrixXf temp(lhs.cols(),rhs.cols());
    //    temp.setZero();
        LOG(INFO)<<"done! ";
        LOG_IF(FATAL,check_matrix_invalid(temp))<<"computation become invalid! fatal wrong!";
        R = temp.transpose();
}

void train3::save_R(const MatrixXf &R, const std::string &file_name)
{
    FILE *file = fopen(file_name.data(),"wb");
    LOG_IF(FATAL,!file)<<"train2::save_R: "<<file_name<<" file open failed";
    int temp = R.rows();
    fwrite(&temp,sizeof(int),1,file);
    temp = R.cols();
    fwrite(&temp,sizeof(int),1,file);
    fwrite(R.data(),sizeof(float),R.size(),file);
    fclose(file);
}

bool train3::check_matrix_invalid(const MatrixXf &matrix)
{
    const float* data = matrix.data();
    for(int i=0;i<matrix.size();i++)
    {
        if(isnan(*data)||isinf(*data))
            return true;
        data++;
    }
    return false;
}

void train3::read_keypos_R(const std::string &readmodel_root, int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name = readmodel_root+"keypos_Rs_"+num.toStdString()+".bin";
    MatrixXf &R=m_keypos_Rs[casscade_level];
    read_R(R,name);
}

void train3::read_para_R(const std::string &readmodel_root, int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name = readmodel_root+"paras_Rs_"+num.toStdString()+".bin";
    MatrixXf &R=m_para_Rs[casscade_level];
    read_R(R,name);
}

void train3::read_R(MatrixXf &R, const std::string &file_name)
{
    FILE *file = fopen(file_name.data(),"rb");
    LOG_IF(FATAL,!file)<<"train3::read_R: "<<file_name<<" file open failed";
    int R_col;
    int R_row;
    fread(&R_row,sizeof(int),1,file);
    fread(&R_col,sizeof(int),1,file);
    R.resize(R_row,R_col);
    fread(R.data(),sizeof(float),R.size(),file);
    fclose(file);
}
