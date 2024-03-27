#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void MaxoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  CHECK_EQ(bottom.size(), 1) << "Maxout layer takes only one input blob.";
		  CHECK_EQ(top->size(), 1) << "Maxout layer takes only one output blob.";
		  kernelSize = this->layer_param_.kernelsize();
		  CHECK_EQ(bottom[0]->channels() % kernelSize, 0) << "The number of input channels should be divided by kernel size.";
		  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels() / kernelSize, bottom[0]->height(), bottom[0]->width());
		  maxIndex.reset(new SyncedMemory((*top)[0]->count() * sizeof(int)));
	}

	template <typename Dtype>
	void MaxoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
		  int num = bottom[0]->num();
		  int outChannels = (*top)[0]->channels();
		  int inDim = bottom[0]->count() / num;
		  int outDim = (*top)[0]->count() / num;
		  int featMapSize = (*top)[0]->count() / num / outChannels;
		  const Dtype* bottom_data = bottom[0]->cpu_data();
		  Dtype* top_data = (*top)[0]->mutable_cpu_data();
		  int* idx_data = (int*)maxIndex->mutable_cpu_data(); 
		  for(int i = 0; i < num; i++)
		  {
			  for(int j = 0; j < featMapSize; j++)
			  { 
				  for(int c = 0; c < outChannels; c++)
				  {
					  Dtype maxAct = bottom_data[i*inDim + c*kernelSize*featMapSize];
					  int maxIdx = 0;
					  for(int k = 1; k < kernelSize; k++)
					  {
						  if( maxAct < bottom_data[i*inDim + (c*kernelSize+k)*featMapSize + j])
						  {
							  maxAct = bottom_data[i*inDim + (c*kernelSize+k)*featMapSize + j];
							  maxIdx = k;
						  }
					  }
					  top_data[i*outDim + c*featMapSize + j] = maxAct;
					  idx_data[i*outDim + c*featMapSize + j] = maxIdx;
				  }
			  }
		  }
	}

	template <typename Dtype>
	Dtype MaxoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
		  int num = (*bottom)[0]->num();
		  int outChannels = top[0]->channels();
		  int inDim = (*bottom)[0]->count() / num;
		  int outDim = top[0]->count() / num;
		  int featMapSize = top[0]->count() / num / outChannels;
		  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		  Dtype* bottom_2nd_diff = (*bottom)[0]->mutable_cpu_2nd_diff();
		  const Dtype* top_diff = top[0]->cpu_diff();
		  const Dtype* top_2nd_diff = top[0]->cpu_2nd_diff();
		  const int* idx_data = (int*)maxIndex->cpu_data(); 
		  memset(bottom_diff, 0, sizeof(Dtype)*(*bottom)[0]->count());
		  for(int i = 0; i < num; i++)
		  {
			  for(int j = 0; j < featMapSize; j++)
			  { 
				  for(int c = 0; c < outChannels; c++)
				  {
					  int maxIdx = idx_data[i*outDim + c*featMapSize + j];
					  bottom_diff[i*inDim + (c*kernelSize+maxIdx)*featMapSize + j] = top_diff[i*outDim + c*featMapSize + j];
					  if(this->layer_param_.cal_2nd_grad())
						  bottom_2nd_diff[i*inDim + (c*kernelSize+maxIdx)*featMapSize + j] = top_2nd_diff[i*outDim + c*featMapSize + j];
				  }
			  }
		  }
		  return Dtype(0);
	}

	INSTANTIATE_CLASS(MaxoutLayer);
}