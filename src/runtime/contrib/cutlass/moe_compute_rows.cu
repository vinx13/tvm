// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/moe_kernels.cu
__device__ inline int find_total_elts_leq_target(const int* sorted_indices, const int arr_length, const int target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high) {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target) {
            high = mid - 1;
        }
        else {
            low             = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}
__global__ void compute_total_rows_before_expert_kernel(const int*    sorted_experts,
                                                        const int     sorted_experts_len,
                                                        const int64_t num_experts,
                                                        int64_t*      total_rows_before_expert)
{

    // First, compute the global tid. We only need 1 thread per expert.
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;

    // This should construct the last index where each expert occurs.
    total_rows_before_expert[expert] = find_total_elts_leq_target(sorted_experts, sorted_experts_len, expert);
}

void compute_total_rows_before_expert(const int*   sorted_indices,
                                      const int    total_indices,
                                      const int    num_experts,
                                      int64_t*     total_rows_before_expert,
                                      cudaStream_t stream)
{

    const int threads = std::min(1024, num_experts);
    const int blocks  = (num_experts + threads - 1) / threads;

    compute_total_rows_before_expert_kernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}
