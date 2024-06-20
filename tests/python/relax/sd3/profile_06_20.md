| Name | Duration (us) | Percent | Device | Count | Argument Shapes |
| --- | --- | --- | --- | --- | --- |
| fused_relax_nn_attention_cutlass1 | 17755.45 | 24.86 | cuda0 | 24 | float16[2, 4250, 24, 64], float16[2, 4250, 24, 64], float16[2, 4250, 24, 64], uint8[26112000], float16[2, 4250, 24, 64] |
| fused_relax_permute_dims_relax_matmul1_cublas | 6635.74 | 9.29 | cuda0 | 96 | float16[1536, 1536], float16[8192, 1536], float16[8192, 1536] |
| fused_relax_permute_dims_relax_matmul_relax_add4_cublas | 5377.94 | 7.53 | cuda0 | 24 | float16[1536, 6144], float16[8192, 6144], float16[1536], float16[8192, 1536] |
| fused_relax_permute_dims_relax_matmul_relax_add_relax_nn_gelu_cublas | 5081.01 | 7.12 | cuda0 | 24 | float16[6144, 1536], float16[8192, 1536], float16[6144], float16[8192, 6144] |
| fused_reshape7_add6_reshape5_add2_concatenate1_reshape9 | 2858.40 | 4.00 | cuda0 | 72 | float16[8192, 1536], float16[1536], float16[308, 1536], float16[1536], float16[2, 4250, 24, 64] |
| fused_relax_nn_layer_norm_cutlass | 1882.50 | 2.64 | cuda0 | 49 | float16[2, 4096, 1536], float16[1536], float16[1536], float16[2, 4096, 1536] |
| fused_relax_permute_dims_relax_matmul2_cublas | 1565.81 | 2.19 | cuda0 | 95 | float16[1536, 1536], float16[308, 1536], float16[308, 1536] |
| fused_expand_dims2_add3_multiply3_expand_dims2_add4_reshape6 | 1556.71 | 2.18 | cuda0 | 48 | float16[2, 1536], float16[2, 1536], float16[2, 4096, 1536], float16[8192, 1536] |
| fused_reshape10_split2 | 1094.00 | 1.53 | cuda0 | 23 | float16[2, 4250, 24, 64], float16[2, 4096, 1536], float16[2, 154, 1536] |
| fused_relax_permute_dims_relax_matmul_relax_add3_cublas | 1050.95 | 1.47 | cuda0 | 47 | float16[9216, 1536], float16[2, 1536], float16[9216], float16[2, 9216] |
| split1 | 868.65 | 1.22 | cuda0 | 47 | float16[2, 9216], float16[2, 1536], float16[2, 1536], float16[2, 1536], float16[2, 1536], float16[2, 1536], float16[2, 1536] |
| fused_reshape7_add6_expand_dims2_multiply5_add7 | 838.36 | 1.17 | cuda0 | 24 | float16[2, 1536], float16[8192, 1536], float16[1536], float16[2, 4096, 1536], float16[2, 4096, 1536] |
| fused_reshape7_expand_dims2_multiply5_add7 | 747.70 | 1.05 | cuda0 | 24 | float16[2, 1536], float16[8192, 1536], float16[2, 4096, 1536], float16[2, 4096, 1536] |
| fused_relax_permute_dims_relax_matmul_relax_add_relax_nn_gelu1_cublas | 654.13 | 0.92 | cuda0 | 23 | float16[6144, 1536], float16[308, 1536], float16[6144], float16[308, 6144] |
| fused_relax_permute_dims_relax_matmul_relax_add5_cublas | 569.27 | 0.80 | cuda0 | 23 | float16[1536, 6144], float16[308, 6144], float16[1536], float16[308, 1536] |
| fused_relax_nn_layer_norm1_cutlass | 366.85 | 0.51 | cuda0 | 47 | float16[2, 154, 1536], float16[1536], float16[1536], float16[2, 154, 1536] |
| fused_expand_dims2_add3_multiply4_expand_dims2_add5_reshape8 | 302.28 | 0.42 | cuda0 | 46 | float16[2, 1536], float16[2, 1536], float16[2, 154, 1536], float16[308, 1536] |
| fused_reshape5_add2_expand_dims2_multiply6_add8 | 168.89 | 0.24 | cuda0 | 23 | float16[2, 1536], float16[308, 1536], float16[1536], float16[2, 154, 1536], float16[2, 154, 1536] |
| fused_reshape5_expand_dims2_multiply6_add8 | 163.70 | 0.23 | cuda0 | 23 | float16[2, 1536], float16[308, 1536], float16[2, 154, 1536], float16[2, 154, 1536] |
| vm.builtin.make_tuple | 151.99 | 0.21 | cuda0 | 47 | float16[2, 1536], float16[2, 1536], float16[2, 1536], float16[2, 1536], float16[2, 1536], float16[2, 1536] |
| vm.builtin.make_tuple | 74.83 | 0.10 | cuda0 | 23 | float16[2, 4096, 1536], float16[2, 154, 1536] |
| vm.builtin.reshape | 74.47 | 0.10 | cuda0 | 23 | float16[2, 4096, 1536] |
| vm.builtin.reshape | 74.25 | 0.10 | cuda0 | 23 | float16[2, 154, 1536] |
| fused_reshape10_split2_reshape6 | 65.50 | 0.09 | cuda0 | 1 | float16[2, 4250, 24, 64], float16[8192, 1536] |
| fused_transpose3_reshape1_transpose4_reshape2_strided_slice_reshape3_add | 32.09 | 0.04 | cuda0 | 1 | float16[2, 64, 64, 1536], float16[1, 36864, 1536], float16[2, 4096, 1536] |
| fused_split3_expand_dims2_add3_multiply3_expand_dims2_add4_reshape6 | 31.20 | 0.04 | cuda0 | 1 | float16[2, 3072], float16[2, 4096, 1536], float16[8192, 1536] |
| fused_relax_permute_dims_relax_matmul_relax_add6_cublas | 27.13 | 0.04 | cuda0 | 2 | float16[3072, 1536], float16[2, 1536], float16[3072], float16[2, 3072] |
| fused_relax_permute_dims_relax_matmul_relax_add1_cublas | 23.84 | 0.03 | cuda0 | 2 | float16[1536, 1536], float16[2, 1536], float16[1536], float16[2, 1536] |
| fused_relax_nn_conv2d_relax_add_cutlass | 21.70 | 0.03 | cuda0 | 1 | float16[2, 128, 128, 16], float16[1536, 2, 2, 16], float16[1, 1, 1, 1536], float16[2, 64, 64, 1536] |
| fused_relax_permute_dims_relax_matmul_cublas | 21.47 | 0.03 | cuda0 | 1 | float16[1536, 4096], float16[308, 4096], float16[308, 1536] |
| fused_relax_permute_dims_relax_matmul3_cublas | 16.57 | 0.02 | cuda0 | 1 | float16[64, 1536], float16[8192, 1536], float16[8192, 64] |
| fused_relax_permute_dims_relax_matmul_relax_add2_cublas | 13.47 | 0.02 | cuda0 | 1 | float16[1536, 2048], float16[2, 2048], float16[1536], float16[2, 1536] |
| silu | 11.07 | 0.02 | cuda0 | 2 | float16[2, 1536], float16[2, 1536] |
| fused_reshape11_add9_reshape12_einsum_reshape13 | 10.78 | 0.02 | cuda0 | 1 | float16[8192, 64], float16[64], float16[2, 16, 128, 128] |
| fused_relax_permute_dims_relax_matmul_relax_add_cublas | 8.67 | 0.01 | cuda0 | 1 | float16[1536, 256], float16[2, 256], float16[1536], float16[2, 1536] |
| transpose | 7.65 | 0.01 | cuda0 | 1 | float16[2, 16, 128, 128], float16[2, 128, 128, 16] |
| fused_split3_expand_dims2_add3_multiply4_expand_dims2_add5_reshape8 | 6.69 | 0.01 | cuda0 | 1 | float16[2, 3072], float16[2, 154, 1536], float16[308, 1536] |
| fused_reshape5_add2 | 6.69 | 0.01 | cuda0 | 1 | float16[308, 1536], float16[1536], float16[2, 154, 1536] |
| transpose1 | 6.21 | 0.01 | cuda0 | 1 | float16[1536, 16, 2, 2], float16[1536, 2, 2, 16] |
| fused_expand_dims_cast_multiply1_multiply2_tir_sin_tir_cos_concatenate_split_concatenate_cast1 | 6.02 | 0.01 | cuda0 | 1 | float16[2], float32[1, 128], float16[2, 256] |
| fused_add1_silu | 5.76 | 0.01 | cuda0 | 1 | float16[2, 1536], float16[2, 1536], float16[2, 1536] |
| vm.builtin.match_shape | 3.20 | 0.00 | cuda0 | 1 | float16[2, 2048] |
| vm.builtin.make_tuple | 3.20 | 0.00 | cuda0 | 1 | float16[2, 16, 128, 128] |
| vm.builtin.match_shape | 3.20 | 0.00 | cuda0 | 1 | float16[2, 16, 128, 128] |
| vm.builtin.check_tensor_info | 3.20 | 0.00 | cuda0 | 1 | float16[2, 2048] |
| vm.builtin.match_shape | 3.20 | 0.00 | cuda0 | 1 | float16[2, 154, 4096] |
| vm.builtin.reshape | 3.20 | 0.00 | cuda0 | 1 | float16[2, 154, 4096] |
| vm.builtin.check_tensor_info | 3.20 | 0.00 | cuda0 | 1 | float16[2, 154, 4096] |
| vm.builtin.check_tensor_info | 3.20 | 0.00 | cuda0 | 1 | float16[2, 16, 128, 128] |
| vm.builtin.reshape | 3.17 | 0.00 | cuda0 | 1 | float16[1536] |
| vm.builtin.check_tensor_info | 3.17 | 0.00 | cuda0 | 1 | float16[2] |
| vm.builtin.match_shape | 3.17 | 0.00 | cuda0 | 1 | float16[2, 1536] |
| **Total Duration (us)** | **50258.73** | **100.00** | - | - | - |
