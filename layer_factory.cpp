// Copyright 2013 Yangqing Jia

#ifndef CAFFE_LAYER_FACTORY_HPP_
#define CAFFE_LAYER_FACTORY_HPP_

#include <string>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "tree/ForestLayer.h"


namespace caffe {


// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  const std::string& type = param.type();
  if (type == "accuracy") {
    return new AccuracyLayer<Dtype>(param);
  } else if (type == "bnll") {
    return new BNLLLayer<Dtype>(param);
  } else if (type == "conv") {
    return new ConvolutionLayer<Dtype>(param);
  } else if (type == "data") {
    return new DataLayer<Dtype>(param);
  } else if (type == "copy") {
	return new CopyLayer<Dtype>(param);
  } else if (type == "merge") {
	return new MergeLayer<Dtype>(param);
  } else if (type == "dropout") {
    return new DropoutLayer<Dtype>(param);
  } else if (type == "euclidean_loss") {
    return new EuclideanLossLayer<Dtype>(param);
  } else if (type == "flatten") {
    return new FlattenLayer<Dtype>(param);
  } else if (type == "forest") {
	return new ForestLayer<Dtype>(param);
  } else if (type == "im2col") {
    return new Im2colLayer<Dtype>(param);
  } else if (type == "infogain_loss") {
    return new InfogainLossLayer<Dtype>(param);
  } else if (type == "innerproduct") {
    return new InnerProductLayer<Dtype>(param);
  } else if (type == "maxout") {
	return new MaxoutLayer<Dtype>(param);
  } else if (type == "lrn") {
    return new LRNLayer<Dtype>(param);
  }	else if (type == "lambda_rank") {
	return new LambdaRankLossLayer<Dtype>(param);
 // } else if (type == "listnet") {
//	return new ListnetLossLayer<Dtype>(param);
  } else if (type == "padding") {
    return new PaddingLayer<Dtype>(param);
  } else if (type == "pool") {
    return new PoolingLayer<Dtype>(param);
  } else if (type == "relu") {
    return new ReLULayer<Dtype>(param);
  } else if (type == "rmse") {
	  return new RmseLayer<Dtype>(param);
  } else if (type == "n_rmse") {
	  return new NRmseLayer<Dtype>(param);
  } else if (type == "ndcg") {
	  return new NdcgLayer<Dtype>(param);
  }else if(type == "sigmoid") {
	  return new SigmoidLayer<Dtype>(param);
  }else if(type == "gaussian") {
	  return new GaussianLayer<Dtype>(param);
  }else if (type == "softmax") {
    return new SoftmaxLayer<Dtype>(param);
  } else if (type == "softmax_loss") {
    return new SoftmaxWithLossLayer<Dtype>(param);
  } else if (type == "klogistic_loss") {
	  return new KLogisticLossLayer<Dtype>(param);
  } else if (type == "cross_entropy") {
	  return new CrossEntropyLayer<Dtype>(param);
  } else if (type == "n_cross_entropy") {
	  return new NCrossEntropyLayer<Dtype>(param);
  } else if (type == "multinomial") {
	  return new MultinomialLayer<Dtype>(param);
  } else if (type == "multinomial_logistic_loss") {
    return new MultinomialLogisticLossLayer<Dtype>(param);
  } else {
    LOG(FATAL) << "Unknown layer name: " << type;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}

template Layer<float>* GetLayer(const LayerParameter& param);
template Layer<double>* GetLayer(const LayerParameter& param);

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_HPP_
