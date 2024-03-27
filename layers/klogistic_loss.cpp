#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

const float kLOG_THRESHOLD = 1e-20;

using std::max;

namespace caffe {

	template <typename Dtype>
	void KLogisticLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	  CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
	  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
		  << "The data and label should have the same dimension and number.";
	};


	template <typename Dtype>
	Dtype KLogisticLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
	  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	  int count = (*bottom)[0]->count();
	  int num = (*bottom)[0]->num();
	  memset(bottom_diff, 0, sizeof(Dtype) * count);
	  Dtype loss = 0;
	  for (int i = 0; i < (*bottom)[0]->count(); ++i) {
		Dtype eNum = max(min(Dtype(45), bottom_data[i]), Dtype(-45));
		Dtype prob = 1 / (1 + exp(-eNum));
		loss = loss - bottom_label[i]*log(prob) - (1-bottom_label[i])*log(1 - prob) ;
		bottom_diff[i] = (prob - bottom_label[i]) / num;
	  }
	  return loss / num;
	}

	INSTANTIATE_CLASS(KLogisticLossLayer);
}