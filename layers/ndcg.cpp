#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/device_vector.h>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "l2r/util.h"

using std::max;

namespace caffe {

	template <typename Dtype>
	void NdcgLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
	  CHECK_EQ(bottom.size(), 3) << "NDCG Layer takes three blobs as input.";
	  CHECK_EQ(top->size(), 1) << "NDCG Layer takes 1 output.";
	  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
		  << "The data and label should have the same number.";
	  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
		  << "The data and qid should have the same number";
	  (*top)[0]->Reshape(1, 1, 1, 1);
	  topK = this->layer_param_.top_k();
	}

	template <typename Dtype>
	void NdcgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		const Dtype* label = bottom[1]->cpu_data();
		const Dtype* scores = bottom[0]->cpu_data();
		const Dtype* qid = bottom[2]->cpu_data();
		vector<int> qNum;
		int curGroupId = qid[0];
		int startIdx = 0;
		for(int i = 0; i < bottom[2]->num(); i++)
		{
			if(qid[i] != curGroupId)
			{
				qNum.push_back(i - startIdx);
				startIdx = i;
				curGroupId = qid[i];
			}
		}
		qNum.push_back(bottom[2]->num() - startIdx);
		Dtype sumNdcg = 0;
		startIdx = 0;

		for(int q = 0; q < qNum.size(); q++)
		{
			sumNdcg += calNdcg(scores+startIdx, label+startIdx, qNum[q], topK);
			startIdx += qNum[q];
		}

		(*top)[0]->mutable_cpu_data()[0] = sumNdcg / qNum.size();
	}

	INSTANTIATE_CLASS(NdcgLayer);
}