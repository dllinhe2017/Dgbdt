#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void CopyLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  CHECK_EQ(bottom.size(), 1) << "Copy layer takes only one input blobs.";
		  for(int i = 0; i < top->size(); i++)
			  (*top)[i]->Reshape(bottom[0]->num(), bottom[0]->channels(),
					bottom[0]->height(), bottom[0]->width());
	}

	template <typename Dtype>
	void CopyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  int count = bottom[0]->count();
		  const Dtype* bottom_data = bottom[0]->cpu_data();
		  for(int i = 0; i < top->size(); i++)
			  caffe_copy(count, bottom_data, (*top)[i]->mutable_cpu_data());
	}

	template <typename Dtype>
	void CopyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  int count = bottom[0]->count();
		  const Dtype* bottom_data = bottom[0]->gpu_data();
		  for(int i = 0; i < top->size(); i++)
			  caffe_gpu_copy(count, bottom_data, (*top)[i]->mutable_gpu_data());
	}

	INSTANTIATE_CLASS(CopyLayer);
}