#include "train.h"
#include "io_utils.h"
#include <QString>
#include "supplement_gpu_math_functions.hpp"
#include "cuda.h"
#include <cuda_runtime.h>
//#include <Eigen/IterativeLinearSolvers>
train::train(int thread_num)
{
    m_data_root = "../../multi_learn_data/clip_data_224/";
    m_savemodel_root = "../save_model/";
    m_para_mean_sd_file="../resource/para_mean_sd.bin";
    m_para_num = 6;
    m_casscade_sum = 5;
    //per face_img, except all imgs to train, random select 10 times random num imgs to join train, add robust
    //used in compute_shapes_exps_R_b()
    per_face_img_random_train_data_num=5;
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

void train::read_img_datas(const std::string &meshpara_list, const std::string &permesh_imglist)
{
    std::vector<std::string> mesh_files;
//    get_meshpara_names(root, mesh_files);
    io_utils::read_all_type_file_to_vector<std::string>(meshpara_list.data(),mesh_files);
    std::vector<std::vector<std::string > > per_imgfiles;
//    get_permesh_imgnames(root, mesh_files, per_imgfiles);
    io_utils::read_all_type_rowsfile_to_2vector<std::string>(permesh_imglist.data(),per_imgfiles);
    m_face_imgs.clear();
    m_all_img_num=0;
    LOG(INFO)<<"start read person imgs...";
    for(int i=0; i<mesh_files.size(); i++)
    {
        face_imgs temp(m_data_root,mesh_files[i],
                       Face::get_shape_pcanum(),Face::get_exp_pcanum());
        temp.set_img_num(per_imgfiles[i]);
        temp.read_imgs();
        temp.read_pose_paras();
        temp.read_shape_exp();
        m_face_imgs.push_back(temp);
        m_all_img_num+=per_imgfiles[i].size();
        LOG_IF(INFO,i%100==99)<<i<<" person imgs have been readed.";
    }
    LOG(INFO)<<"read "<<mesh_files.size()<<" person imgs "<<"done!";

    //initial need by multithread feature computation variables
    m_face_imgs_pointer.clear();
    m_before_imgsize.clear();
    std::list<face_imgs>::iterator iter=m_face_imgs.begin();
    int before_imgsize=0;
    for(;iter!=m_face_imgs.end();iter++)
    {
        m_face_imgs_pointer.push_back(&(*iter));
        m_before_imgsize.push_back(before_imgsize);
        before_imgsize+=iter->img_size();
    }

    m_face_img_num = m_face_imgs.size();
    m_train_paras.resize(m_para_num,m_all_img_num);
    m_train_shapes.resize(Face::get_shape_pcanum(),m_face_img_num);
    m_train_exps.resize(Face::get_exp_pcanum(),m_face_img_num);
    m_visible_features.resize(Face::get_keypoints_size()*m_feature_size,m_all_img_num);

//    compute_para_mean_sd();
//    save_para_mean_sd();
}

void train::train_model()
{
    initial_shape_exp_with_mean();
    initial_para();
    m_casscade_level=-1;
    show_delta_para_info();
    show_delta_shape_exp_info();
    m_para_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    m_para_bs.resize(m_casscade_sum, Eigen::VectorXf());

    m_shape_exp_Rs.resize(m_casscade_sum, Eigen::MatrixXf());

    for(m_casscade_level=0; m_casscade_level<m_casscade_sum; m_casscade_level++)
    {
        compute_all_visible_features_multi_thread();
        compute_paras_R_b();
        update_para();
        show_delta_para_info();
        save_para_result(m_casscade_level);
        //learn id and exp
        compute_all_visible_features_multi_thread();
////divide id and exp to train
//        compute_shapes_R_b();
//        update_shape();
//        show_delta_shape_info();

//        compute_all_visible_features();
//        compute_exps_R_b();
//        update_exp();
//        show_delta_exp_info();

        //combine shape and exp to train
        compute_shapes_exps_R_b();
        update_shape_exp(per_face_img_random_train_data_num);
        show_delta_shape_exp_info();
        save_shape_exp_result(m_casscade_level);
    }
}

void train::verify_model()
{
    initial_shape_exp_with_mean();
    initial_para();
    m_casscade_level=-1;
    show_delta_para_info();
    show_delta_shape_exp_info();

    for(m_casscade_level=0; m_casscade_level<m_casscade_sum; m_casscade_level++)
    {
        compute_all_visible_features_multi_thread();
        update_para();
        show_delta_para_info();
        //learn id and exp
        compute_all_visible_features_multi_thread();
        //only use a face all imgs to verify
        compute_regular_features(0);
        update_shape_exp(0);
        show_delta_shape_exp_info();
    }
}

void train::save_verify_result(const std::string &root)
{
    for(int i=0;i<m_face_imgs_pointer.size();i++)
    {
        face_imgs *data = m_face_imgs_pointer[i];
        std::string file=root+data->get_face_name()+"_mesh_para.txt";
        Eigen::VectorXf temp(Face::get_shape_pcanum()+Face::get_exp_pcanum());
        temp.block(0,0,Face::get_shape_pcanum(),1) = m_train_shapes.col(i);
        temp.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),1) = m_train_exps.col(i);
        io_utils::write_all_type_to_file<float>(temp,file);
        const std::vector<std::string> &names=data->get_imgfiles_name();
        for(int j=0;j<data->img_size();j++)
        {
            int id = m_before_imgsize[i]+j;
            QString tname(names[j].data());
            tname.remove(temp.size()-4,4);
            file = root+tname.toStdString()+"_pose.txt";
            io_utils::write_all_type_to_file<float>(m_train_paras.col(id),file);
        }
    }
}

void train::read_trained_model(const std::string &root, int casscade_num)
{
    m_casscade_sum = casscade_num;
    m_para_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    m_para_bs.resize(m_casscade_sum, Eigen::VectorXf());
    m_shape_exp_Rs.resize(m_casscade_sum, Eigen::MatrixXf());
    for(int i=0;i<casscade_num;i++)
    {
        read_para_result(root,i);
        read_shape_exp_result(root,i);
    }
}

void train::save_para_result(int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name=m_savemodel_root+"para_Rs_"+num.toStdString()+".bin";
    FILE *file = fopen(name.data(),"wb");
    LOG_IF(FATAL,!file)<<"save_para_result "<<name<<" file open failed";
    //save b with R
    int R_col = Face::get_keypoints_size()*m_feature_size+1;
    int R_row = m_para_num;
    fwrite(&R_row,sizeof(int),1,file);
    fwrite(&R_col,sizeof(int),1,file);
    fwrite(m_para_Rs[casscade_level].data(),sizeof(float),(R_col-1)*R_row,file);
    fwrite(m_para_bs[casscade_level].data(),sizeof(float),R_row,file);
    fclose(file);
}

void train::save_shape_exp_result(int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name = m_savemodel_root+"shape_exp_Rs_"+num.toStdString()+".bin";
    FILE *file = fopen(name.data(),"wb");
    LOG_IF(FATAL,!file)<<"save_shape_exp_result "<<name<<" file open failed";
    //save b with R
    int R_col = (m_feature_size*Face::get_keypoints_size()+1)*m_orien_choose.get_divide_num();
    int R_row = Face::get_shape_pcanum()+Face::get_exp_pcanum();
    fwrite(&R_row,sizeof(int),1,file);
    fwrite(&R_col,sizeof(int),1,file);
    fwrite(m_shape_exp_Rs[casscade_level].data(),sizeof(float),R_col*R_row,file);
    fclose(file);
}

void train::set_feature_compute_gpu(const std::vector<int> ids)
{
    m_feature_detectors.clear();
    if(ids.size()!=m_threadnum_for_compute_features)
    {
        LOG(FATAL)<<"train::set_feature_compute_gpu ids size wrong";
        m_gpuid_for_feature_computes.resize(1,0);
        m_feature_detectors.push_back(CNNDenseFeature(0));
    }
    m_gpuid_for_feature_computes=ids;
    for(int i=0;i<m_threadnum_for_compute_features;i++)
    {
        m_feature_detectors.push_back(CNNDenseFeature(m_gpuid_for_feature_computes[i]));
        LOG(INFO)<<"Net "<<i<<" has been build.";
    }
}

void train::initial_shape_exp_with_mean()
{
    m_train_shapes.setZero();
    m_train_exps.setZero();
}

void train::initial_para()
{
//    std::vector<float> paras;
//    io_utils::read_all_type_file_to_vector<float>("../resource/mean_para_file.txt", paras);
//    Eigen::VectorXf temp(6);
    for(int i=0; i<m_all_img_num; i++)
        m_train_paras.col(i) = m_groundtruth_paras_mean;
}

void train::compute_para_mean_sd()
{
    m_groundtruth_paras_sd.resize(m_para_num);
    m_groundtruth_paras_mean.resize(m_para_num);
    m_groundtruth_paras_mean.setZero();
    Eigen::MatrixXf temp(m_para_num,m_all_img_num);
    std::list<face_imgs>::iterator iter = m_face_imgs.begin();
    int index=0;
    for(; iter!=m_face_imgs.end(); iter++)
    {
        temp.block(0,index,m_para_num,iter->img_size()) = iter->get_groundtruth_paras();
        for(int id=0; id<iter->img_size(); id++)
        {
            m_groundtruth_paras_mean += iter->get_groundtruth_para(id);
            index++;
        }

    }
    m_groundtruth_paras_mean /= m_all_img_num;
    temp = temp.colwise()-m_groundtruth_paras_mean;
    temp /= sqrt(float(m_all_img_num-1));
    m_groundtruth_paras_sd=temp.rowwise().norm();
}

void train::read_para_mean_sd()
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

void train::save_para_mean_sd()
{
    std::vector<float> vals(2*m_para_num, 0);
    memcpy(vals.data(),m_groundtruth_paras_mean.data(),m_para_num*sizeof(float));
    memcpy(vals.data()+m_para_num,m_groundtruth_paras_sd.data(),m_para_num*sizeof(float));
    io_utils::write_all_type_to_bin<float>(vals,"../resource/para_mean_sd.bin");
}

void train::compute_shapes_exps_mean_sd()
{
    m_groundtruth_shapes_exps_sd.resize(Face::get_shape_pcanum()+Face::get_exp_pcanum());
    m_groundtruth_shapes_exps_sd.block(0,0,Face::get_shape_pcanum(),1)=Face::get_shape_st();
    m_groundtruth_shapes_exps_sd.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),1)=Face::get_exp_st();
}

void train::compute_delta_para(MatrixXf &delta_para)
{
    delta_para.resize(m_para_num,m_all_img_num);
    std::list<face_imgs>::iterator iter = m_face_imgs.begin();
    int index=0;
    for(; iter!=m_face_imgs.end(); iter++)
    {
        for(int id=0; id<iter->img_size(); id++)
        {
            delta_para.col(index) = iter->get_groundtruth_para(id)-m_train_paras.col(index);
            index++;
        }
    }
}

void train::compute_delta_shape_exp(MatrixXf &delta_para)
{
    delta_para.resize(Face::get_shape_pcanum()+Face::get_exp_pcanum(),m_face_img_num);
    std::list<face_imgs>::iterator iter = m_face_imgs.begin();
    int index=0;
    for(; iter!=m_face_imgs.end(); iter++)
    {
        delta_para.block(0,index,Face::get_shape_pcanum(),1) = iter->get_groundtruth_shapes()-m_train_shapes.col(index);
        delta_para.block(Face::get_shape_pcanum(),index,Face::get_exp_pcanum(),1) = iter->get_groundtruth_exps()-m_train_exps.col(index);
        index++;
    }
}

//void train::compute_all_visible_features()
//{
//    std::list<face_imgs>::iterator iter = m_face_imgs.begin();
//    int index=0;
//    int img_size = 0;
//    for(; iter!=m_face_imgs.end(); iter++)
//    {
//        compute_per_face_imgs_visiblefeatures(iter,index,img_size);
//        img_size+=iter->img_size();
//        index++;
//        if(index%500==0)
//            std::cout<<index<<" face img features have been computed"<<std::endl;
//    }
//}

void train::compute_all_visible_features_multi_thread()
{
    Eigen::VectorXi nums(m_threadnum_for_compute_features);
    nums.setZero();
    LOG(INFO)<<"start compute visible features";
    #pragma omp parallel for num_threads(m_threadnum_for_compute_features)
    for(int i=0;i<m_face_imgs_pointer.size();i++)
    {

        int thread_id=omp_get_thread_num();
        //for Caffe set thread independent Caffe setting, this is essential, otherwise non-major thread caffe will
        //not take the setting on CNNDenseFeature, like setmode(GPU), still default CPU
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(m_gpuid_for_feature_computes[thread_id]);


        compute_per_face_imgs_visiblefeatures_multi_thread(m_face_imgs_pointer[i],i,m_before_imgsize[i],thread_id);
        nums[thread_id]=nums[thread_id]+1;
        LOG_IF(INFO,nums.sum()%500==0)<<"face img features have been computed a 500";
    }
    LOG(INFO)<<"done!";
}

void train::compute_per_face_imgs_visiblefeatures_multi_thread(face_imgs *iter, int index, int img_size, int thread_id)
{
    for(int i=0;i<iter->img_size();i++)
    {
        int col = img_size+i;
        float *para = m_train_paras.col(col).data();
        float scale = para[0];
        float ax = para[1]; float ay = para[2]; float az = para[3];
        float tx = para[4]; float ty = para[5];

        Eigen::Affine3f transformation;
        transformation  = Eigen::AngleAxisf(ax, Eigen::Vector3f::UnitX()) *
                          Eigen::AngleAxisf(ay, Eigen::Vector3f::UnitY()) *
                          Eigen::AngleAxisf(az, Eigen::Vector3f::UnitZ());
        Eigen::Matrix3f R = transformation.rotation();
//        m_3dmm_mesh.set_shape(iter->get_groundtruth_shapes());
//        m_3dmm_mesh.set_exp(iter->get_groundtruth_exps());
        m_3dmm_meshs[thread_id].set_shape(m_train_shapes.col(index));
        m_3dmm_meshs[thread_id].set_exp(m_train_exps.col(index));
//        m_3dmm_mesh.update_mesh(false);
        Eigen::MatrixXf temp_v;
        m_3dmm_meshs[thread_id].get_vertex_matrix(temp_v,false);
        temp_v  = R*temp_v;
        temp_v *=scale;
        Eigen::Vector3f trans;  trans(0) = tx;  trans(1) = ty;  trans(2) = 0.0;
        temp_v.colwise() += trans;
        Eigen::MatrixXf verts(3, Face::get_keypoints_size());
        for(int j=0;j<Face::get_keypoints_size();j++)
        {
            verts.col(j) = temp_v.col(Face::get_part_keypoints()[j]);
        }
        Eigen::MatrixXf feature_pos(2,Face::get_keypoints_size()) ;
        feature_pos = verts.block(0,0,2,Face::get_keypoints_size());
        //convert to left up origin
        for(int j=0;j<Face::get_keypoints_size();j++)
            feature_pos(1,j) = iter->img_length()-feature_pos(1,j);
        //visibles compute; can change to use openGL, more accurate
        std::vector<bool>   visuals;
        compute_keypoint_visible_multi_thread(R,visuals,thread_id);
        // add extract feature code, para: grayImage feature_pos visuals, result:  visible_features; need to compute visible
        Eigen::VectorXf visible_features(m_feature_size*Face::get_keypoints_size());
//        Eigen::VectorXf scales(m_keypoint_id.size()); scales.setOnes();
//        scales *= 6.0;
//        if(!m_feature_detector->DescriptorOnCustomPoints(image,visuals,feature_pos,scales,visible_features) )
//            std::cout<<"----------------feature computation has some wrong-----------------------"<<std::endl;

        m_feature_detectors[thread_id].set_data(iter->get_img(i));
        m_feature_detectors[thread_id].get_compute_visible_posfeatures(feature_pos,visuals,visible_features);

        m_visible_features.col(col) = visible_features;
    }
}

void train::compute_keypoint_visible_multi_thread(const Eigen::MatrixXf &R, std::vector<bool> &visuals, int thread_id)
{
    //this mesh's coordinate have not times Rotation
    const TriMesh& temp_mesh = m_3dmm_meshs[thread_id].get_part_face();
    visuals.resize(Face::get_keypoints_size(), false);
//    TriMesh::Normal zdir(0.0,0.0,1.0);
    for(int i=0; i< Face::get_keypoints_size(); i++)
    {
        int id = Face::get_part_keypoints()[i];
        TriMesh::Normal normal = temp_mesh.normal(TriMesh::VertexHandle(id));
        Eigen::Vector3f Rnormal = Eigen::Map<Eigen::Vector3f>(normal.data());
        Rnormal =R*Rnormal;
        float val = Rnormal(2);
        if(val>0.0)
        {
            visuals[i] = true;
////whether in image range check move to feature detecture to check!
//            TriMesh::Point p = temp_mesh.point(TriMesh::VertexHandle(id));
//            //check inside img
//            int x = p[0]+0.5;
//            int y = p[1]+0.5;
//            if(x<width&&x>=0&&y<height&&y>=0)
//                visuals[i] = true;
        }
    }
}

void train::compute_paras_R_b()
{
    float lamda = 100;
    Eigen::MatrixXf delta_x;
    compute_delta_para(delta_x);
    //normalize paras
    delta_x = (1.0/m_groundtruth_paras_sd.array()).matrix().asDiagonal()*delta_x;

    //combine R and b to solve, new_R=(R|b)
//    const int R_row = m_para_num;
    const int R_col = Face::get_keypoints_size()*m_feature_size+1;
    Eigen::MatrixXf lhs(R_col,R_col);
    Eigen::MatrixXf rhs(R_col,m_para_num);

//    lhs.block(0,0,R_col-1,R_col-1).selfadjointView<Eigen::Upper>().rankUpdate(m_visible_features,1.0);
    //using gpu to compute rankUpdate
    Eigen::MatrixXf rankTemp;
    my_gpu_rankUpdated(rankTemp,m_visible_features,1.0,m_gpuid_for_matrix_compute);
    lhs.block(0,0,R_col-1,R_col-1)=rankTemp;


    lhs.block(0,R_col-1,R_col-1,1) = m_visible_features.rowwise().sum();
    lhs(R_col-1,R_col-1) = float(m_all_img_num);
    //add regular
    lhs.block(0,0,R_col-1,R_col-1)+=(lamda*Eigen::VectorXf(R_col-1).setOnes()).asDiagonal();

    rhs.block(0,0,R_col-1,m_para_num) = m_visible_features*delta_x.transpose();
    rhs.bottomRows(1) = delta_x.transpose().colwise().sum();

    lhs.triangularView<Eigen::Lower>() = lhs.transpose();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" compute for A("<<lhs.rows()<<"*"<<lhs.cols()<<")...";
//using Dense decompose
    Eigen::MatrixXf temp = lhs.ldlt().solve(rhs);
//    //using ConjugateGradient method
//    Eigen::ConjugateGradient<Eigen::MatrixXf,Lower|Upper> cg;
//    cg.compute(lhs);
//    Eigen::MatrixXf temp=cg.solve(rhs);

    LOG_IF(FATAL,check_matrix_invalid(temp))<<"computation become invalid! fatal wrong!";
    Eigen::MatrixXf result = temp.transpose();
    LOG(INFO)<<"done! ";
    Eigen::MatrixXf &R = m_para_Rs[m_casscade_level];
    R.resize(m_para_num,m_feature_size*Face::get_keypoints_size());
    memcpy(R.data(),result.data(), sizeof(float)*R.size());
    Eigen::VectorXf &b = m_para_bs[m_casscade_level];
    b.resize(m_para_num);
    b = result.col(R_col-1);
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" result norm: "<<result.norm()<<"; sqrt energy is "<<(lhs*result.transpose()-rhs).norm()<<
        " delta norm: "<<(((m_para_Rs[m_casscade_level]*m_visible_features).colwise()+m_para_bs[m_casscade_level])-delta_x).colwise().norm().mean();
}

void train::update_para()
{
    Eigen::MatrixXf delta_x;
    delta_x = (m_para_Rs[m_casscade_level]*m_visible_features).colwise()+m_para_bs[m_casscade_level];
    //unnormalize paras
    delta_x = m_groundtruth_paras_sd.asDiagonal()*delta_x;

    m_train_paras+=delta_x;
}

void train::show_delta_para_info()
{
    Eigen::MatrixXf delta_para;
    compute_delta_para(delta_para);
    Eigen::MatrixXf delta_s = delta_para.row(0);
    Eigen::MatrixXf delta_r = delta_para.block(1,0,3,m_all_img_num);
    Eigen::MatrixXf delta_t = delta_para.block(4,0,2,m_all_img_num);
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

void train::compute_regular_feature_from_multi_imgs(int before_img_size,const std::vector<int> &choose_imgs_ids, MatrixXf &regu_features)
{
    std::vector<int> sub_space_num(m_orien_choose.get_divide_num(),0);
    regu_features.resize((m_feature_size*Face::get_keypoints_size()+1)*m_orien_choose.get_divide_num(),1);
    regu_features.setZero();
    for(int i=0;i<choose_imgs_ids.size(); i++)
    {
        int col = before_img_size+choose_imgs_ids[i];
        float * arc=m_train_paras.col(col).data();
        float ax = arc[1];
        float ay = arc[2];
        float az = arc[3];
        int id=m_orien_choose.compute_orientation_id(ax,ay,az);
        sub_space_num[id]=sub_space_num[id]+1;

        regu_features.block(id*(m_feature_size*Face::get_keypoints_size()+1),0,m_feature_size*Face::get_keypoints_size(),1)
                += m_visible_features.col(col);
        regu_features(id*(m_feature_size*Face::get_keypoints_size()+1)+m_feature_size*Face::get_keypoints_size(),0)
                += 1.0;
    }
    int zero_num=0;
    for(int i=0;i<sub_space_num.size();i++)
        if(sub_space_num[i]==0)
            zero_num++;
    for(int i=0;i<sub_space_num.size();i++)
    {
        if(sub_space_num[i]!=0)
        {
            float ratio = 1.0/float(sub_space_num[i]*(sub_space_num.size()-zero_num));
            regu_features.block(i*(m_feature_size*Face::get_keypoints_size()+1),0,m_feature_size*Face::get_keypoints_size()+1,1)
                    *= ratio;
        }
    }
}

void train::compute_regular_features(int randomnum)
{
    m_regu_features.resize((m_feature_size*Face::get_keypoints_size()+1)*m_orien_choose.get_divide_num()
                           ,m_face_img_num*(1+randomnum));

    std::list<face_imgs>::iterator iter = m_face_imgs.begin();
    int before_img_size=0;
    int index=0;
    std::vector<int> choose_ids;
    Eigen::MatrixXf regu_features;
    for(; iter!=m_face_imgs.end(); iter++)
    {
        //first face img's all img to train
        choose_ids.clear();
        for(int i=0;i<iter->img_size();i++)
            choose_ids.push_back(i);
        compute_regular_feature_from_multi_imgs(before_img_size,choose_ids,regu_features);
        m_regu_features.col((1+randomnum)*index) = regu_features;
        //is per face_img add other random choose img to  train
        if(randomnum>0)
        {
            for(int i=0;i<randomnum;i++)
            {
                int random_num=m_random_tool.random_choose_num_from_1_to_size(iter->img_size());
                m_random_tool.random_choose_n_different_id_from_0_to_size(random_num,iter->img_size()-1,choose_ids);
                compute_regular_feature_from_multi_imgs(before_img_size,choose_ids,regu_features);
                m_regu_features.col((1+randomnum)*index+i+1) = regu_features;
            }
        }
        before_img_size+=iter->img_size();
        index++;
    }
}

void train::compute_shapes_exps_R_b()
{
    float lamda = 20;
    Eigen::MatrixXf delta_x;
    compute_delta_shape_exp(delta_x);
    //normalize paras
    delta_x = (1.0/m_groundtruth_shapes_exps_sd.array()).matrix().asDiagonal()*delta_x;

    Eigen::MatrixXf expand_delta_x;
    if(per_face_img_random_train_data_num==0)
        expand_delta_x = delta_x;
    else
        expand_delta_x_with_num(per_face_img_random_train_data_num, delta_x, expand_delta_x);
    compute_regular_features(per_face_img_random_train_data_num);
    //combine R and b to solve, new_R=(R|b)
    const int R_col = m_regu_features.rows();
    Eigen::MatrixXf lhs(R_col,R_col);
    Eigen::MatrixXf rhs(R_col,expand_delta_x.cols());

//    lhs.selfadjointView<Eigen::Upper>().rankUpdate(m_regu_features,1.0);
    //using gpu to compute rankUpdate
    my_gpu_rankUpdated(lhs,m_regu_features,1.0,m_gpuid_for_matrix_compute);

    //add regular
    lhs+=(lamda*Eigen::VectorXf(R_col).setOnes()).asDiagonal();

    rhs = m_regu_features*expand_delta_x.transpose();

    lhs.triangularView<Eigen::Lower>() = lhs.transpose();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" compute for pca para A("<<lhs.rows()<<"*"<<lhs.cols()<<")...";

//    io_utils::write_all_type_to_bin<float>(lhs.data(),"lhs_matrix.bin",lhs.size(),true);
//    io_utils::write_all_type_to_bin<float>(rhs.data(),"rhs_matrix.bin",rhs.size(),true);

    //using Dense decompose method
    Eigen::MatrixXf temp = lhs.ldlt().solve(rhs);

//    //using ConjugateGradient method
//    Eigen::ConjugateGradient<Eigen::MatrixXf,Lower|Upper> cg;
//    cg.compute(lhs);
//    Eigen::MatrixXf temp=cg.solve(rhs);

    LOG_IF(FATAL,check_matrix_invalid(temp))<<"computation become invalid! fatal wrong!";

    m_shape_exp_Rs[m_casscade_level]=temp.transpose();
    LOG(INFO)<<"done! ";
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" result norm: "<<m_shape_exp_Rs[m_casscade_level].norm()<<"; sqrt energy is "<<(lhs*temp-rhs).norm()<<
            " delta norm: "<<(m_shape_exp_Rs[m_casscade_level]*m_regu_features-expand_delta_x).colwise().norm().mean();
}

void train::update_shape_exp(int num)
{
    Eigen::MatrixXf expand_delta_x,delta_x;
    expand_delta_x = m_shape_exp_Rs[m_casscade_level]*m_regu_features;
    depand_delta_x_with_num(num,expand_delta_x,delta_x);
    //unnormalize paras
    delta_x = m_groundtruth_shapes_exps_sd.asDiagonal()*delta_x;

    m_train_shapes+=delta_x.block(0,0,Face::get_shape_pcanum(),m_face_img_num);
    m_train_exps+=delta_x.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),m_face_img_num);
}

void train::show_delta_shape_exp_info()
{
    Eigen::MatrixXf delta_para;
    compute_delta_shape_exp(delta_para);
    Eigen::MatrixXf delta_shape = delta_para.block(0,0,Face::get_shape_pcanum(),m_face_img_num);
    Eigen::MatrixXf delta_exp = delta_para.block(Face::get_shape_pcanum(),0,Face::get_exp_pcanum(),m_face_img_num);
    Eigen::MatrixXf norm;
    norm = (delta_shape.row(0)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" shape 0 with ground truth norm: mean: "<<norm.mean()
            <<" max: "<<norm.maxCoeff()<<" min: "<<norm.minCoeff();
    norm = (delta_shape.row(1)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" shape 1 with ground truth norm: mean: "<<norm.mean()
            <<" max: "<<norm.maxCoeff()<<" min: "<<norm.minCoeff();
    norm = (delta_shape.row(2)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" shape 2 with ground truth norm: mean: "<<norm.mean()
            <<" max: "<<norm.maxCoeff()<<" min: "<<norm.minCoeff();

    norm = (delta_exp.row(0)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" exp 0 with ground truth norm: mean: "<<norm.mean()
            <<" max: "<<norm.maxCoeff()<<" min: "<<norm.minCoeff();
    norm = (delta_exp.row(1)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" exp 1 with ground truth norm: mean: "<<norm.mean()
            <<" max: "<<norm.maxCoeff()<<" min: "<<norm.minCoeff();
    norm = (delta_exp.row(2)).colwise().norm();
    LOG(INFO)<<"casscade para "<<m_casscade_level<<" exp 2 with ground truth norm: mean: "<<norm.mean()
            <<" max: "<<norm.maxCoeff()<<" min: "<<norm.minCoeff();
}

void train::expand_delta_x_with_num(int num, const MatrixXf &delta_x, MatrixXf &expand_delta_x)
{
    expand_delta_x.resize(delta_x.rows(),delta_x.cols()*(1+num));
    for(int i=0;i<delta_x.cols();i++)
    {
        for(int j=i*(1+num);j<(i+1)*(1+num);j++)
            expand_delta_x.col(j)=delta_x.col(i);
    }
}

void train::depand_delta_x_with_num(int num, const MatrixXf &expand_delta_x, MatrixXf &delta_x)
{
    delta_x.resize(expand_delta_x.rows(),expand_delta_x.cols()/(1+num));
    for(int i=0;i<delta_x.cols();i++)
    {
        delta_x.col(i)=expand_delta_x.col(i*(1+num));
    }
}

bool train::check_matrix_invalid(const MatrixXf &matrix)
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

void train::read_para_result(const std::string &readmodel_root,int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name=readmodel_root+"para_Rs_"+num.toStdString()+".bin";
    FILE *file = fopen(name.data(),"rb");
    LOG_IF(FATAL,!file)<<"read_para_result "<<name<<" file open failed";
    //read b with R
    int R_col;
    int R_row;
    fread(&R_row,sizeof(int),1,file);
    fread(&R_col,sizeof(int),1,file);
    m_para_Rs[casscade_level].resize(R_row,R_col-1);
    m_para_bs[casscade_level].resize(R_row,1);
    fread(m_para_Rs[casscade_level].data(),sizeof(float),(R_col-1)*R_row,file);
    fread(m_para_bs[casscade_level].data(),sizeof(float),R_row,file);
    fclose(file);
}

void train::read_shape_exp_result(const std::string &readmodel_root, int casscade_level)
{
    QString num;
    num.setNum(casscade_level);
    std::string name = readmodel_root+"shape_exp_Rs_"+num.toStdString()+".bin";
    FILE *file = fopen(name.data(),"rb");
    LOG_IF(FATAL,!file)<<"read_shape_exp_result "<<name<<" file open failed";
    //read b with R
    int R_col;
    int R_row;
    fread(&R_row,sizeof(int),1,file);
    fread(&R_col,sizeof(int),1,file);
    m_shape_exp_Rs[casscade_level].resize(R_row,R_col);
    fread(m_shape_exp_Rs[casscade_level].data(),sizeof(float),R_col*R_row,file);
    fclose(file);
}

void train::my_gpu_rankUpdated(MatrixXf &C, const MatrixXf &A, float a, int gpu_id)
{
    LOG(INFO)<<"start rankUpdate...";
    LOG_IF(FATAL, A.size()==0)<<"train::my_gpu_rankUpdated input matrix are empty!";
    void *C_data;
    void *A_data;
    int C_size=A.rows()*A.rows()*sizeof(float);
    int A_size=A.size()*sizeof(float);
    Caffe::SetDevice(gpu_id);
    CUDA_CHECK(cudaGetDevice(&gpu_id));
    CUDA_CHECK(cudaMalloc(&C_data, C_size));
    caffe_gpu_memset(C_size, 0, C_data);

    CUDA_CHECK(cudaMalloc(&A_data, A_size));
    caffe_gpu_memcpy(A_size, A.data(), A_data);

    gpu_rankUpdate((float*)C_data,(float*)A_data,A.rows(),A.cols(),a);

    void *temp_cpu_C_data;
    bool cpu_malloc_use_cuda;
    CaffeMallocHost(&temp_cpu_C_data, C_size, &cpu_malloc_use_cuda);
    caffe_gpu_memcpy(C_size, C_data, temp_cpu_C_data);

    C.resize(A.rows(),A.rows());
    memcpy(C.data(),temp_cpu_C_data,C_size);

    CUDA_CHECK(cudaFree(C_data));
    CUDA_CHECK(cudaFree(A_data));
    CaffeFreeHost(temp_cpu_C_data,cpu_malloc_use_cuda);
    LOG(INFO)<<"done";
}
