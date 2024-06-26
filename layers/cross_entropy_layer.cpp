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
void CrossEntropyLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "CrossEntropy Layer takes 2 blob as input.";
  CHECK_EQ(top->size(), 0) << "CrossEntropy Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
};

template <typename Dtype>
void CrossEntropyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype CrossEntropyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // First, compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  for (int i = 0; i < prob_.count(); ++i) {
    bottom_diff[i] -= label[i];
    loss -= label[i] * log(max(prob_data[i], Dtype(FLT_MIN)));
  }

  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  return loss / num;
}

INSTANTIATE_CLASS(CrossEntropyLayer);

template <typename Dtype>
void NCrossEntropyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
	CrossEntropyLayer<Dtype>::Forward_cpu(bottom, top);
	Dtype* prob_data = prob_.mutable_cpu_data();
	const Dtype* label_data = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / num;
	for(int i = 0; i < num; i++)
	{
		Dtype scale = 0;
		for(int j = 0; j < dim; j++)
			scale += label_data[i*dim + j];
		caffe_scal(dim, scale, prob_data + i*dim);
	}
}

INSTANTIATE_CLASS(NCrossEntropyLayer);

}  // namespace caffe
