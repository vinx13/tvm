/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file Conv2d backward in cuDNN
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.backward_data")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int pad_h = args[3];
  int pad_w = args[4];
  int stride_h = args[5];
  int stride_w = args[6];
  int dilation_h = args[7];
  int dilation_w = args[8];
  DLTensor *x = args[9];
  DLTensor *w = args[10];
  DLTensor *y = args[11];
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  // entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set Ctx
  entry_ptr->conv_entry.ctx = x->ctx;
  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Set Desc
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             entry_ptr->conv_entry.mode,
                                             entry_ptr->conv_entry.data_type));
  // Set Filter
  CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                        entry_ptr->conv_entry.data_type,
                                        CUDNN_TENSOR_NCHW,
                                        static_cast<int>(w->shape[0]),
                                        static_cast<int>(w->shape[1]),
                                        static_cast<int>(w->shape[2]),
                                        static_cast<int>(w->shape[3])));
  // Set Input
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(x->shape[0]),
                                        static_cast<int>(x->shape[1]),
                                        static_cast<int>(x->shape[2]),
                                        static_cast<int>(x->shape[3])));
  // Set Output
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.output_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(y->shape[0]),
                                        static_cast<int>(y->shape[1]),
                                        static_cast<int>(y->shape[2]),
                                        static_cast<int>(y->shape[3])));
  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(entry_ptr->handle,
                                                     entry_ptr->conv_entry.filter_desc,
                                                     entry_ptr->conv_entry.output_desc,
                                                     entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.input_desc,
                                                     CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                                     &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);
  CUDNN_CALL(cudnnConvolutionBackwardData(entry_ptr->handle,
                                     CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
                                     entry_ptr->conv_entry.filter_desc,
                                     w->data,
                                     entry_ptr->conv_entry.output_desc,
                                     y->data,
                                     entry_ptr->conv_entry.conv_desc,
                                                     CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                     entry_ptr->conv_entry.workspace,
                                     workspace_size,
                                     CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
                                     entry_ptr->conv_entry.input_desc,
                                     x->data));
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.backward_filter")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int pad_h = args[3];
  int pad_w = args[4];
  int stride_h = args[5];
  int stride_w = args[6];
  int dilation_h = args[7];
  int dilation_w = args[8];
  DLTensor *x = args[9];
  DLTensor *w = args[10];
  DLTensor *y = args[11];
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  // entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set Ctx
  entry_ptr->conv_entry.ctx = x->ctx;
  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Set Desc
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             entry_ptr->conv_entry.mode,
                                             entry_ptr->conv_entry.data_type));
  // Set Filter
  CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                        entry_ptr->conv_entry.data_type,
                                        CUDNN_TENSOR_NCHW,
                                        static_cast<int>(w->shape[0]),
                                        static_cast<int>(w->shape[1]),
                                        static_cast<int>(w->shape[2]),
                                        static_cast<int>(w->shape[3])));
  // Set Input
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(x->shape[0]),
                                        static_cast<int>(x->shape[1]),
                                        static_cast<int>(x->shape[2]),
                                        static_cast<int>(x->shape[3])));
  // Set Output
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.output_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(y->shape[0]),
                                        static_cast<int>(y->shape[1]),
                                        static_cast<int>(y->shape[2]),
                                        static_cast<int>(y->shape[3])));
  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(entry_ptr->handle,
                                                     entry_ptr->conv_entry.input_desc,
                                                     entry_ptr->conv_entry.output_desc,
                                                     entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.filter_desc,
                                                     CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                                     &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);
  CUDNN_CALL(cudnnConvolutionBackwardFilter(entry_ptr->handle,
                                     CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
                                     entry_ptr->conv_entry.input_desc,
                                     x->data,
                                     entry_ptr->conv_entry.output_desc,
                                     y->data,
                                     entry_ptr->conv_entry.conv_desc,
                                     CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                     entry_ptr->conv_entry.workspace,
                                     workspace_size,
                                     CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
                                     entry_ptr->conv_entry.filter_desc,
                                     w->data));
});

}  // namespace contrib
}  // namespace tvm
