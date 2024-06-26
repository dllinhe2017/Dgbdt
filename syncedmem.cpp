// Copyright 2013 Yangqing Jia

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_) {
    CaffeFreeHost(cpu_ptr_);
	cpu_ptr_ = NULL;
  }

  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
	gpu_ptr_ = NULL;
  }
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    memset(cpu_ptr_, 0, size_);
    head_ = HEAD_AT_CPU;
    break;
  case HEAD_AT_GPU:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
    }
    CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDeviceToHost));
    head_ = SYNCED;
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice));
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

const void* SyncedMemory::gpu_data() {
  to_gpu();
  return (const void*)gpu_ptr_;
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
}


}  // namespace caffe

