#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/device_vector.h>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

	template <typename Dtype>
	void RmseLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
	  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	  CHECK_EQ(top->size(), 1) << "Loss Layer takes 1 output.";
	  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
		  << "The data and label should have the same number.";
	  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
	  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	  difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
		  bottom[0]->height(), bottom[0]->width());
	  (*top)[0]->Reshape(1, 1, 1, 1);
	}

	template <typename Dtype>
	void RmseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		int count = bottom[0]->count();
		int num = bottom[0]->num();
		caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
			difference_.mutable_cpu_data());
		Dtype rmse = sqrt(caffe_cpu_dot(
			count, difference_.cpu_data(), difference_.cpu_data()) / count);
		(*top)[0]->mutable_cpu_data()[0] = rmse;
	}

	INSTANTIATE_CLASS(RmseLayer);

	template <typename Dtype>
	void NRmseLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		scaleProb.Reshape(bottom[0]->num(), bottom[0]->channels(),
		  bottom[0]->height(), bottom[0]->width());
		rmseLayer->SetUp(bottom, top);
	}

	template <typename Dtype>
	void NRmseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		int num = bottom[0]->num();
		int dim = bottom[0]->count() / num;
		const Dtype* prob_data = bottom[0]->cpu_data();
		Dtype* scale_prob_data = scaleProb.mutable_cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();
		memcpy(scale_prob_data, prob_data, sizeof(Dtype)*bottom[0]->count());
		for(int i = 0; i < num; i++)
		{
			Dtype scale = 0;
			for(int j = 0; j < dim; j++)
				scale += label_data[i*dim + j];
			caffe_scal(dim, scale, scale_prob_data + i*dim);
		}
		rmseBottomVec.push_back(&scaleProb);
		rmseBottomVec.push_back(bottom[1]);
		rmseLayer->Forward(rmseBottomVec, top);
	}

	INSTANTIATE_CLASS(NRmseLayer);

}