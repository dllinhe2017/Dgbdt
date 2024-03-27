#include <vector>
#include <cstdio>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void MergeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  CHECK_EQ(top->size(), 1) << "Merge layer take only one blobs as output.";
		  int totalDim = 0;
		  for(int i = 0; i < bottom.size(); i++)
			  totalDim += bottom[i]->count() / bottom[i]->num();
		  (*top)[0]->Reshape(bottom[0]->num(), totalDim, 1, 1);
	}

	template <typename Dtype>
	void MergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  int num = (*top)[0]->num();
		  int dim = (*top)[0]->count() / num;
		  Dtype* top_data = (*top)[0]->mutable_cpu_data();
		  vector<int> bottom_dim;
		  for(int i = 0; i < bottom.size(); i++)
			  bottom_dim.push_back(bottom[i]->count() / bottom[i]->num());

	      int offset = 0;
	      for(int j = 0; j < bottom.size(); j++)
		  {
			  const Dtype* bottom_data = bottom[j]->cpu_data();
			  for(int i = 0; i < num; i++)
				  caffe_copy(bottom_dim[j], bottom_data+i*bottom_dim[j], top_data+i*dim+offset);
			  offset += bottom_dim[j];
		  }
	}

	template <typename Dtype>
	void MergeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  int num = (*top)[0]->num();
		  int dim = (*top)[0]->count() / num;
		  Dtype* top_data = (*top)[0]->mutable_gpu_data();
		  vector<int> bottom_dim;
		  for(int i = 0; i < bottom.size(); i++)
			  bottom_dim.push_back(bottom[i]->count() / bottom[i]->num());
		  
		 int offset = 0;
		 for(int j = 0; j < bottom.size(); j++)
		 {
			  const Dtype* bottom_data = bottom[j]->gpu_data();
			  for(int i = 0; i < num; i++)
				  caffe_gpu_copy(bottom_dim[j], bottom_data+i*bottom_dim[j], top_data+i*dim+offset);
			  offset += bottom_dim[j];	
		  }
	}

	template <typename Dtype>
	Dtype MergeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		  int num = top[0]->num();
		  int dim = top[0]->count() / num;
		  const Dtype* top_diff = top[0]->cpu_diff();
		  vector<int> bottom_dim;
		  for(int i = 0; i < bottom->size(); i++)
			  bottom_dim.push_back((*bottom)[i]->count() / (*bottom)[i]->num());
		  
		  int offset = 0;
		  for(int j = 0; j < bottom->size(); j++)
		  {
			   Dtype* bottom_diff = (*bottom)[j]->mutable_cpu_diff();
			   for(int i = 0; i < num; i++)
				  caffe_copy(bottom_dim[j], top_diff+i*dim+offset, bottom_diff+i*bottom_dim[j]);
			   offset += bottom_dim[j];
		  }
		  return Dtype(0);
	}

	template <typename Dtype>
	Dtype MergeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		  int num = top[0]->num();
		  int dim = top[0]->count() / num;
		  const Dtype* top_diff = top[0]->gpu_diff();
		  vector<int> bottom_dim;
		  for(int i = 0; i < bottom->size(); i++)
			  bottom_dim.push_back((*bottom)[i]->count() / (*bottom)[i]->num());
		  
		  int offset = 0;
		  for(int j = 0; j < bottom->size(); j++)
		  {
			   Dtype* bottom_diff = (*bottom)[j]->mutable_gpu_diff();
			   for(int i = 0; i < num; i++)
				  caffe_gpu_copy(bottom_dim[j], top_diff+i*dim+offset, bottom_diff+i*bottom_dim[j]);
			   offset += bottom_dim[j];
		  }
		  return Dtype(0);
	}

	INSTANTIATE_CLASS(MergeLayer);
}