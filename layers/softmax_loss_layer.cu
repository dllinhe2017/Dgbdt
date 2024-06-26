// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
};

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // First, compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  if(this->layer_param_.cal_2nd_grad()) {
	  Dtype* bottom_2nd_diff = (*bottom)[0]->mutable_cpu_2nd_diff();
	  for(int i = 0; i < prob_.count(); i++) {
		  bottom_2nd_diff[i] = (1 - prob_data[i]) * prob_data[i];
	  }
  }
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])], FLT_MIN));
  }
  // Scale down gradient
  //caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  return loss / num;
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  // TODO(Yangqing): implement the GPU version of softmax.
  return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
