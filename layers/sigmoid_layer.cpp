#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {
	template <typename Dtype>
	void SigmoidLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		CHECK_EQ(bottom.size(), 1) << "Sigmoid Layer takes a single blob as input.";
	    CHECK_EQ(top->size(), 1) << "Sigmoid Layer takes a single blob as output.";
	    (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		    bottom[0]->height(), bottom[0]->width());
	    ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
		    bottom[0]->height(), bottom[0]->width());
	    Dtype* ones_data = ones_.mutable_cpu_data();
	    for(int i = 0; i < ones_.count(); i++)
		    ones_data[i] = 1;
	}

	template <typename Dtype>
	void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = (*top)[0]->mutable_cpu_data();
		const Dtype* ones_data = ones_.cpu_data();
		int count = bottom[0]->count();
		/*memcpy(top_data, bottom_data, sizeof(Dtype) * count);
		caffe_exp(count, top_data, top_data);
		caffe_div(count, ones_data, top_data, top_data);
		caffe_add(count, ones_data, top_data, top_data);
		caffe_div(count, ones_data, top_data, top_data);*/
		for(int i = 0; i < count; i++)
		{
			float eNum = max(min(Dtype(45), bottom_data[i]), Dtype(-45));
			top_data[i] = 1 / (1 + exp(-eNum));
		}
	}

	template <typename Dtype>
	Dtype SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* ones_data = ones_.cpu_data();
		int count = (*bottom)[0]->count(); 
		/*memcpy(bottom_diff, top_diff, sizeof(Dtype) * count);
		caffe_mul(count, bottom_diff, top_data, bottom_diff);
		caffe_sub(count, ones_data, top_data, top_data);
		caffe_mul(count, bottom_diff, top_data, bottom_diff);*/
		caffe_sub(count, ones_data, top_data, bottom_diff);
		caffe_mul(count, bottom_diff, top_data, bottom_diff);
		caffe_mul(count, bottom_diff, top_diff, bottom_diff);
		return Dtype(0);
	}

	INSTANTIATE_CLASS(SigmoidLayer);
}