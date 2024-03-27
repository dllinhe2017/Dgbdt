#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void GaussianLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		CHECK_EQ(bottom.size(), 1) << "Gaussian Layer takes a single blob as input.";
	    CHECK_EQ(top->size(), 1) << "Gaussian Layer takes a single blob as output.";
	    (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		    bottom[0]->height(), bottom[0]->width());
		delta = this->layer_param_.delta();
	}

	template <typename Dtype>
	void GaussianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = (*top)[0]->mutable_cpu_data();
		int count = bottom[0]->count();
		for(int i = 0; i < count; i++) 
		{
			top_data[i] = exp(-bottom_data[i]*bottom_data[i] / delta /delta);
		}
	}

	template <typename Dtype>
	Dtype GaussianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* top_data = top[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		int count = (*bottom)[0]->count();
		caffe_mul(count, bottom_data, top_data, bottom_diff);
		caffe_mul(count, bottom_diff, top_diff, bottom_diff);
		caffe_scal(count, static_cast<Dtype>(-2/delta/delta), bottom_diff);
		return Dtype(0);
	}

	INSTANTIATE_CLASS(GaussianLayer);
}