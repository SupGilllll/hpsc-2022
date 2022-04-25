#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
    const int N = 8;
    float x[N], y[N], m[N], fx[N], fy[N];
    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        __m256 neg = _mm256_set1_ps(-1.f);
        __m256 xvec_i = _mm256_set1_ps(x[i]);    __m256 xvec_j = _mm256_load_ps(x);
        __m256 yvec_i = _mm256_set1_ps(y[i]);    __m256 yvec_j = _mm256_load_ps(y);
        __m256 rvec_x = _mm256_sub_ps(xvec_i, xvec_j);    __m256 rvec_y = _mm256_sub_ps(yvec_i, yvec_j);
        __m256 ones = _mm256_set1_ps(1.f);    __m256 zeros = _mm256_set1_ps(0.f);
        __m256 mask = _mm256_cmp_ps(xvec_i, xvec_j, _CMP_NEQ_UQ);
        __m256 r = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rvec_x, rvec_x), _mm256_mul_ps(rvec_y, rvec_y)));
        r = _mm256_blendv_ps(zeros, r, mask);
        __m256 mxvec = _mm256_load_ps(m);
        __m256 fvec_x = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(rvec_x, mxvec), _mm256_mul_ps(r, _mm256_mul_ps(r, r))), neg);
        __m256 fvec_y = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(rvec_y, mxvec), _mm256_mul_ps(r, _mm256_mul_ps(r, r))), neg);
        __m256 x_sum = _mm256_permute2f128_ps(fvec_x, fvec_x, 1);
        x_sum = _mm256_add_ps(x_sum, fvec_x);
        x_sum = _mm256_hadd_ps(x_sum, x_sum);
        x_sum = _mm256_hadd_ps(x_sum, x_sum);
        __m256 y_sum = _mm256_permute2f128_ps(fvec_y, fvec_y, 1);
        y_sum = _mm256_add_ps(y_sum, fvec_y);
        y_sum = _mm256_hadd_ps(y_sum, y_sum);
        y_sum = _mm256_hadd_ps(y_sum, y_sum);
        fx[i] = _mm256_cvtss_f32(x_sum);
        fy[i] = _mm256_cvtss_f32(y_sum);
        printf("%d %g %g\n", i, fx[i], fy[i]);
    }
}