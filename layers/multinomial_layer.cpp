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
	void MultinomialLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		CHECK_EQ(bottom.size(), 1) << "Multinomial Layer takes a single blob as input.";
	    CHECK_EQ(top->size(), 1) << "Multinomial Layer takes a single blob as output.";
	    power = this->layer_param_.power();
		CHECK_GE(power, 1) << "Power should be greater than 1";
		int outChannel = bottom[0]->channels() * power; 
		(*top)[0]->Reshape(bottom[0]->num(), outChannel, bottom[0]->height(), bottom[0]->width());
		tmp.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
	}

	template <typename Dtype>
	void MultinomialLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		int num = bottom[0]->num();
		int channel = bottom[0]->channels();
		int featMapDim = bottom[0]->count() / num / channel;
		int bottomDim = bottom[0]->count() / num;
		int topDim = (*top)[0]->count() / num;
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = (*top)[0]->mutable_cpu_data();

		for(int i = 0; i < num; i++)
		{
			for(int j = 0; j < channel; j++)
			{
				caffe_copy<Dtype>(featMapDim, bottom_data+i*bottomDim+j*featMapDim,
					top_data+i*topDim+j*featMapDim*power);
				for(int k = 1; k < power; k++)
				{
					caffe_mul<Dtype>(featMapDim, bottom_data+i*bottomDim+j*featMapDim, 
						top_data+i*topDim+j*featMapDim*power+(k-1)*featMapDim,
						top_data+i*topDim+j*featMapDim*power+k*featMapDim);
				}
			}
		}
	}

	template <typename Dtype>
	void MultinomialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		int num = bottom[0]->num();
		int channel = bottom[0]->channels();
		int featMapDim = bottom[0]->count() / num / channel;
		int bottomDim = bottom[0]->count() / num;
		int topDim = (*top)[0]->count() / num;
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = (*top)[0]->mutable_gpu_data();
		
		for(int i = 0; i < num; i++)
		{
			for(int j = 0; j < channel; j++)
			{
				caffe_gpu_copy<Dtype>(featMapDim, bottom_data+i*bottomDim+j*featMapDim,
					top_data+i*topDim+j*featMapDim*power);
				for(int k = 1; k < power; k++)
				{
					caffe_gpu_mul<Dtype>(featMapDim, bottom_data+i*bottomDim+j*featMapDim, 
						top_data+i*topDim+j*featMapDim*power+(k-1)*featMapDim,
						top_data+i*topDim+j*featMapDim*power+k*featMapDim);
				}
			}
		}
	}

	template <typename Dtype>
	Dtype MultinomialLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		int num = (*bottom)[0]->num();
		int channel = (*bottom)[0]->channels();
		int featMapDim = (*bottom)[0]->count() / num / channel;
		int bottomDim = (*bottom)[0]->count() / num;
		int topDim = top[0]->count() / num;
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* top_data = top[0]->cpu_data();
		Dtype* tmp_data = tmp.mutable_cpu_data();

		for(int i = 0; i < num; i++)
		{
			for(int j = 0; j < channel; j++)
			{
				caffe_copy<Dtype>(featMapDim, top_diff+i*topDim+j*featMapDim*power, 
					bottom_diff+i*bottomDim+j*featMapDim);
				for(int k = 1; k < power; k++)
				{
					caffe_copy<Dtype>(featMapDim, top_data+i*topDim+j*featMapDim*power+(k-1)*featMapDim, tmp_data);
					caffe_mul<Dtype>(featMapDim, tmp_data, 
						top_diff+i*topDim+j*featMapDim*power+k*featMapDim,
						tmp_data);
					caffe_axpy<Dtype>(featMapDim, k+1, tmp_data, bottom_diff+i*bottomDim+j*featMapDim);
				}
			}
		}
		return Dtype(0);
	}

	template <typename Dtype>
	Dtype MultinomialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		int num = (*bottom)[0]->num();
		int channel = (*bottom)[0]->channels();
		int featMapDim = (*bottom)[0]->count() / num / channel;
		int bottomDim = (*bottom)[0]->count() / num;
		int topDim = top[0]->count() / num;
		Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* top_data = top[0]->gpu_data();
		Dtype* tmp_data = tmp.mutable_gpu_data();

		for(int i = 0; i < num; i++)
		{
			for(int j = 0; j < channel; j++)
			{
				caffe_gpu_copy<Dtype>(featMapDim, top_diff+i*topDim+j*featMapDim*power, 
					bottom_diff+i*bottomDim+j*featMapDim);
				for(int k = 1; k < power; k++)
				{
					caffe_gpu_copy<Dtype>(featMapDim, top_data+i*topDim+j*featMapDim*power+(k-1)*featMapDim, tmp_data);
					caffe_gpu_mul<Dtype>(featMapDim, tmp_data, 
						top_diff+i*topDim+j*featMapDim*power+k*featMapDim,
						tmp_data);
					caffe_gpu_axpy<Dtype>(featMapDim, k+1, tmp_data, bottom_diff+i*bottomDim+j*featMapDim);
				}
			}
		}
		return Dtype(0);
	}

	INSTANTIATE_CLASS(MultinomialLayer);
}