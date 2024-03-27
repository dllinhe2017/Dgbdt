#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <functional>
#include <thrust/device_vector.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "l2r/util.h"

#define MAX_EXP_POWER 45
#define EPS 1e-12

using std::max;
using std::min;

namespace caffe {

	template <typename Dtype>
	void LambdaRankLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top)
	{
		CHECK_EQ(bottom.size(), 3) << "Listnet loss Layer takes three blobs as input.";
	    CHECK_EQ(top->size(), 0) << "Listnet loss Layer takes no blobs as output.";
		delta = this->layer_param_.delta();
		topK = this->layer_param_.top_k();
	}

	template <typename Dtype>
	Dtype  LambdaRankLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom)
	{
		Dtype* f_diff = (*bottom)[0]->mutable_cpu_diff();
		Dtype* f_2nd_diff = this->layer_param_.cal_2nd_grad() ? (*bottom)[0]->mutable_cpu_2nd_diff() : NULL;
		const Dtype* label = (*bottom)[1]->cpu_data();
		const Dtype* scores = (*bottom)[0]->cpu_data();
		const Dtype* qid = (*bottom)[2]->cpu_data();
		vector<int> qNum;
		int curGroupId = qid[0];
		int startIdx = 0;
		int dataNum =  (*bottom)[0]->num();
		for(int i = 0; i < dataNum; i++)
		{
			if(qid[i] != curGroupId)
			{
				qNum.push_back(i - startIdx);
				startIdx = i;
				curGroupId = qid[i];
			}
		}
		qNum.push_back(dataNum - startIdx);
		Dtype loss = 0;
		//int dim = *max_element(qNum.begin(), qNum.end());
		//Dtype* changes = new Dtype[dim*dim];
	
		startIdx = 0;

		memset(f_diff, 0, sizeof(Dtype) * dataNum);
		if(f_2nd_diff)
			memset(f_2nd_diff, 0, sizeof(Dtype) * dataNum);

		for(int q = 0; q < qNum.size(); q++)
		{
			int num = qNum[q];
			int k = (topK == 0) ? num : topK;
			//ndcgSwapChg(scores+startIdx, label+startIdx, changes, num, k, dim);
			loss += calNdcg(scores+startIdx, label+startIdx, num, 3);
			vector<Dtype> iRel(label+startIdx, label+startIdx+num);
			sort(iRel.begin(), iRel.end(), greater<Dtype>());
			Dtype maxDcg = calDcg(iRel, k);
			vector<int> perm;
			for(int i = 0; i < num; i++)
				perm.push_back(i);
			ScoreComp<Dtype>::instance->reset(scores+startIdx, label+startIdx);
			sort(perm.begin(), perm.end(), ScoreComp<Dtype>::comp);

			for(int i = 0; i < num; i++)
			{
				//Dtype lambda = 0;
				if(label[startIdx+perm[i]] == 0)
					continue;
				for(int j = 0; j < num; j++)
				{
					//Dtype grad = 0;
					/*if(label[startIdx+i] > label[startIdx+j])
					{
						//grad = -delta / (1 + exp(power));
						grad = -rho;
					}
					else if(label[startIdx+i] < label[startIdx+j])
					{
						//grad = delta / (1 + exp(-power));
						grad = 1 - rho;
					}
					if(f_2nd_diff && label[startIdx+i] != label[startIdx+j])
						f_2nd_diff[startIdx+i] += delta * delta * rho * (1-rho) * changes[i*dim+j];
					lambda += delta * grad * changes[i*dim+j];*/
					/*if(label[startIdx+i] > label[startIdx+j])
					{
						Dtype power = min(Dtype(MAX_EXP_POWER), (Dtype)max(Dtype(-MAX_EXP_POWER), delta * (scores[startIdx+i] - scores[startIdx+j])));
						Dtype rho = 1.0 / (1 + exp(power));
						Dtype swapChg = maxDcg > 0 ? abs((discount[)
						f_diff[startIdx+i] -=  rho * changes[i*dim+j];
						f_diff[startIdx+j] +=  rho * changes[i*dim+j];
						Dtype deltaWeight = rho * (1.0 - rho) * changes[i*dim+j];
						f_2nd_diff[startIdx+i] += deltaWeight;
						f_2nd_diff[startIdx+j] += deltaWeight;
					}*/
					if(label[startIdx+perm[i]] > label[startIdx+perm[j]])
					{
						Dtype power = min(Dtype(MAX_EXP_POWER), (Dtype)max(Dtype(-MAX_EXP_POWER), delta * (scores[startIdx+perm[i]] - scores[startIdx+perm[j]])));
						Dtype rho = 1.0 / (1 + exp(power));
						Dtype swapChg = maxDcg > 0 ? abs((discount[i] - discount[j]) * (gain[int(label[startIdx+perm[i]]+0.5)] - gain[int(label[startIdx+perm[j]]+0.5)]) / maxDcg) : 0;
						f_diff[startIdx+perm[i]] -=  rho * swapChg;//changes[i*dim+j];
						f_diff[startIdx+perm[j]] +=  rho * swapChg;//changes[i*dim+j];
						if(f_2nd_diff)
						{
							Dtype deltaWeight = rho * (1.0 - rho) * swapChg;//changes[i*dim+j];
							f_2nd_diff[startIdx+perm[i]] += deltaWeight;
							f_2nd_diff[startIdx+perm[j]] += deltaWeight;
						}
					}
				}
				//f_diff[startIdx+i] = lambda;
			}
			startIdx += qNum[q];
		}
		/*for(int i = 0; i < dataNum; i++)
		{
			scores[i];
			label[i];
		}*/
		//caffe_scal((*bottom)[0]->count(), (Dtype)1./dataNum, f_diff);
		//delete[] changes;
		loss /= qNum.size();
		return loss ;
	}

	INSTANTIATE_CLASS( LambdaRankLossLayer);
}