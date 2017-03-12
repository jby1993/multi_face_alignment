#include "cnndensefeature.h"
#include "io_utils.h"
CNNDenseFeature::CNNDenseFeature(int gpu_id, bool normalized)
{
    is_normalized = normalized;
    f_length=224;   //normalized image size
    f_dimension=64;
    img_size=f_length*f_length;
    is_updated = true;
//    mean_image.resize(img_size,0.0);
    //this is left down origin, the feature need be left up origine
    std::vector<float> temp_mean_img;
    io_utils::read_all_type_from_bin<float>("../resource/meanimgfile.bin",img_size,temp_mean_img);
    mean_image.resize(img_size,0.0);
    for(int row=0;row<f_length;row++)
    {
        for(int col=0;col<f_length;col++)
        {
            mean_image[row*f_length+col] = temp_mean_img[(f_length-row-1)*f_length+col];
        }
    }


    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(gpu_id);
// feature extraction net
    features.resize(img_size*f_dimension,0.0);
    construct_net("../resource/feature_extraction_net.prototxt","../resource/feature_extraction_net.caffemodel");
////classfication result net
//    features.resize(img_size*500,0.0);
//    construct_net("../resource/feature_extraction_net2.prototxt","../resource/feature_extraction_net.caffemodel");
}

void CNNDenseFeature::set_data(const std::vector<float> &data)
{
    if(data.size()!=img_size)
    {
        LOG(FATAL)<<"CNNDenseFeature::set_data data size is wrong!";
        return;
    }
    BlobProto blob_proto;
    blob_proto.mutable_shape()->clear_dim();
    blob_proto.mutable_shape()->add_dim(1);
    blob_proto.mutable_shape()->add_dim(1);
    blob_proto.mutable_shape()->add_dim(f_length);
    blob_proto.mutable_shape()->add_dim(f_length);
    blob_proto.clear_data();
    for(int i=0;i<img_size;i++)
        blob_proto.add_data(data[i]-mean_image[i]);
    feature_net->input_blobs()[0]->FromProto(blob_proto);
//    //ce shi fen lei xiao guo de classindex shuru
//    blob_proto.mutable_shape()->clear_dim();
//    blob_proto.mutable_shape()->add_dim(1);
//    blob_proto.mutable_shape()->add_dim(1);
//    blob_proto.clear_data();
//    blob_proto.add_data(99);
//    feature_net->input_blobs()[1]->FromProto(blob_proto);

    is_updated = false;
}

const std::vector<float> &CNNDenseFeature::get_features()
{
    if(!is_updated)
        feature_compute();
    return features;
}

void CNNDenseFeature::get_compute_visible_posfeatures(const Eigen::MatrixXf &pos, const std::vector<bool> &visible, Eigen::VectorXf &visible_features)
{
    if(!is_updated)
        feature_compute();
    visible_features.resize(f_dimension*visible.size());
    if(pos.cols()!=visible.size())
    {
        LOG(FATAL)<<"CNNDenseFeature::get_compute_visible_posfeatures pos visible size is wrong!";
        visible_features.setZero();
        return;
    }
    for(int i=0;i<visible.size();i++)
    {
        if(visible[i])
        {
            int x = pos(0,i)+0.5;
            int y = pos(1,i)+0.5;
            if(x<0||x>=f_length||y<0||y>=f_length)
            {
                visible_features.block(f_dimension*i,0,f_dimension,1).setZero();
                continue;
            }            
//            //no nomalize
//            for(int id=0;id<f_dimension;id++)
//                visible_features(i*f_dimension+id) = features[id*img_size+y*f_length+x];
            //nomalize
            if(is_normalized)
            {
                Eigen::VectorXf temp(f_dimension);
                for(int id=0;id<f_dimension;id++)
                    temp(id) = features[id*img_size+y*f_length+x];
                if(temp.norm()<1.e-6)
                    temp.setZero();
                else
                    temp = temp/temp.norm();
                visible_features.block(i*f_dimension,0,f_dimension,1) = temp;
            }
            else
            {
                for(int id=0;id<f_dimension;id++)
                    visible_features(i*f_dimension+id) = features[id*img_size+y*f_length+x];
            }
        }
        else
        {
            visible_features.block(f_dimension*i,0,f_dimension,1).setZero();
        }
    }
}

void CNNDenseFeature::feature_compute()
{
    float loss=0;
    const std::vector<Blob<float>*>& result=feature_net->Forward(&loss);
    if(result[0]->count()!=features.size())
    {
        LOG(FATAL)<<"CNNDenseFeature::feature_compute result feature size is wrong!";
        return;
    }
    caffe_copy<float>(features.size(),result[0]->cpu_data(), features.data());
//    caffe_copy<float>(features.size(),features.data(), features.data());
    is_updated=true;
}

void CNNDenseFeature::construct_net(std::string model, std::string weights)
{
    feature_net=caffe::shared_ptr<Net<float> >(new Net<float>(model, caffe::TEST));
    feature_net->CopyTrainedLayersFrom(weights);
}
