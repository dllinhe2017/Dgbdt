// Copyright 2013 Yangqing Jia


//#include <mkl.h>
#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";
  const int num_output = this->layer_param_.num_output();
  biasterm_ = this->layer_param_.biasterm();
  // Figure out the dimensions
  M_ = bottom[0]->num();
  K_ = bottom[0]->count() / bottom[0]->num();
  N_ = num_output;
  (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (biasterm_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, N_, K_));
	if(this->layer_param_.cal_2nd_grad())
	{
		sq_weight_.reset(new SyncedMemory(N_ * K_ * sizeof(Dtype)));
		sq_input_.reset(new SyncedMemory(bottom[0]->count() * sizeof(Dtype)));
	}
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(
        GetFiller<Dtype>(this->layer_param_.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (biasterm_) {
      this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, N_));
      shared_ptr<Filler<Dtype> > bias_filler(
          GetFiller<Dtype>(this->layer_param_.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  // Setting up the bias multiplier
  if (biasterm_) {
    bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data =
        reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < M_; ++i) {
        bias_multiplier_data[i] = 1.;
    }
  }
};

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (biasterm_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_2nd_diff = this->layer_param_.cal_2nd_grad() ? top[0]->cpu_2nd_diff() : NULL;
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  // Gradient with respect to weight
   //Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1./M_,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_cpu_diff());
  //caffe_scal(this->blobs_[0]->count(), (Dtype)1./M_, this->blobs_[0]->mutable_cpu_diff());
  if(top_2nd_diff)
  {
	  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
	  Dtype* weight_2nd_diff = this->blobs_[0]->mutable_cpu_2nd_diff();
	  Dtype* sq_input = reinterpret_cast<Dtype*>(sq_input_->mutable_cpu_data());
	  caffe_mul((*bottom)[0]->count(), bottom_data, bottom_data, sq_input);
	  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1./M_, 
		  top_2nd_diff, sq_input, (Dtype)0., weight_2nd_diff);
	  //for(int i = 0; i < this->blobs_[0]->count(); i++)
		  //weight_diff[i] = weight_diff[i] / (abs(weight_2nd_diff[i]) + FLT_EPSILON);
  }
  if (biasterm_) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1./M_, top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
        this->blobs_[1]->mutable_cpu_diff());
	if(top_2nd_diff)
	{
		Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
		Dtype* bias_2nd_diff = this->blobs_[1]->mutable_cpu_2nd_diff();
		caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1./M_, top_2nd_diff,
			reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), (Dtype)0.,
			bias_2nd_diff);
		//for(int i = 0; i < this->blobs_[1]->count(); i++)
			//bias_diff[i] = bias_diff[i] / (abs(bias_2nd_diff[i]) + FLT_EPSILON);
	}
  }
  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_cpu_diff());
	if(top_2nd_diff)
	{
		Dtype* sq_weight = reinterpret_cast<Dtype*>(sq_weight_->mutable_cpu_data());
		Dtype* weight_data= this->blobs_[0]->mutable_cpu_data();
		caffe_mul(this->blobs_[0]->count(), weight_data, weight_data, sq_weight);
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
			top_2nd_diff, sq_weight, (Dtype)0.,
			(*bottom)[0]->mutable_cpu_2nd_diff());
	}
  }
  return Dtype(0);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (biasterm_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  // Gradient with respect to weight
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  if (biasterm_) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)0., this->blobs_[1]->mutable_gpu_diff());
	CUDA_CHECK(cudaDeviceSynchronize()); 
  }
  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_gpu_diff());
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
