#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void sort(int* key, int* bucket)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = 0;
    if (i < 5)
    {
        bucket[i] = 0;
    }
    __syncthreads();
    atomicAdd(&bucket[key[i]], 1);
    __syncthreads();
    if (i < 5)
    {
        for (int k = i; k > 0; --k)
            j += bucket[k - 1];
        for (; bucket[i] > 0; bucket[i]--) {
            key[j++] = i;
        }
    }
}

int main() {
    const int N = 50;
    int n = 50;
    int range = 5;
    int* key, * bucket;
    cudaMallocManaged(&key, n * sizeof(int));
    cudaMallocManaged(&bucket, range * sizeof(int));
    //std::vector<int> key(n);
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    sort <<<1, N>>> (key, bucket);
    cudaDeviceSynchronize();
    //std::vector<int> bucket(range);
    //for (int i = 0; i < range; i++) {
    //    bucket[i] = 0;
    //}
    //for (int i = 0; i < n; i++) {
    //    bucket[key[i]]++;
    //}
    //for (int i = 0, j = 0; i < range; i++) {
    //    for (; bucket[i] > 0; bucket[i]--) {
    //        key[j++] = i;
    //    }
    //}
    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");
    cudaFree(key);
    cudaFree(bucket);
}