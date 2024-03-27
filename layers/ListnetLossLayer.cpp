#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "l2r/util.h"

using std::max;
using std::min;

namespace caffe {
	template <typename Dtype>
	void ListnetLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		CHECK_EQ(bottom.size(), 2) << "Listnet loss Layer takes two blobs as input.";
	    CHECK_EQ(top->size(), 0) << "Listnet loss Layer takes no blobs as output.";
		p_y.Reshape(1, bottom[0]->num(), 1, 1);
		p_f.Reshape(1, bottom[0]->num(), 1, 1);
	}

	template <typename Dtype>
	void ListnetLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		num = 0;
		while(((int)bottom[1]->cpu_data()[num]) >= 0)
			num++;
		const Dtype* f_data = bottom[0]->cpu_data();
		const Dtype* label_data = bottom[1]->cpu_data();

		Dtype* p_y_data = p_y.mutable_cpu_data();
		Dtype exp_y_sum = 0;
		for(int i = 0; i < num; i++)
		{
			p_y_data[i] = exp(label_data[i]);
			exp_y_sum += p_y_data[i];
		}
		caffe_scal(num, 1/exp_y_sum, p_y_data);

		Dtype* p_f_data = p_f.mutable_cpu_data();
		Dtype exp_f_sum = 0;
		for(int i = 0; i < num; i++)
		{
			p_f_data[i] = exp(f_data[i]);
			exp_f_sum += p_f_data[i];
		}
		caffe_scal(num, 1/exp_f_sum, p_f_data);
	}

	template <typename Dtype>
	Dtype ListnetLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		Dtype* f_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* p_f_data = p_f.cpu_data();
		const Dtype* p_y_data = p_y.cpu_data();
		caffe_sub(num, p_f_data, p_y_data, f_diff);
		Dtype loss = 0;
		for(int i = 0; i < num; i++)
			loss -= p_y_data[i] * log(p_f_data[i]);
		return loss / num;
	}

	INSTANTIATE_CLASS(ListnetLossLayer);
}