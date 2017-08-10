#include <stdio.h>
#include <mkl.h>
#include <sys/time.h>
#include "param.h"

#define rep(i, n) for (auto i = static_cast<decltype(n)>(0); i < (n); ++i)

void img2col(const float *src, float *dst,
         size_t /* OC */,  size_t OH,  size_t OW,
         size_t IC,  size_t IH,  size_t IW,
         size_t FH,  size_t FW, bool is_xcorr)
{
    size_t offset = (4 - OW % 4) % 4;
    size_t i = 0;
    rep(ic, IC) {
        rep(fh, FH) {
            rep(fw, FW) {
                rep(oh, OH) {
                    size_t ow = 0;
                    for (; ow < OW; ow += 4) {
                        size_t fh2, fw2;
                        if (is_xcorr) {
                            fh2 = fh;
                            fw2 = fw;
                        } else {
                            fh2 = FH-fh-1;
                            fw2 = FW-fw-1;
                        }
                        dst[i++] = src[ic*IH*IW + (oh+fh2)*IW + (ow+fw2) + 0];
                        dst[i++] = src[ic*IH*IW + (oh+fh2)*IW + (ow+fw2) + 1];
                        dst[i++] = src[ic*IH*IW + (oh+fh2)*IW + (ow+fw2) + 2];
                        dst[i++] = src[ic*IH*IW + (oh+fh2)*IW + (ow+fw2) + 3];
                    }
                    i -= offset;
                }
            }
        }
    }
}


void Im2colMKL(float* input, float* filter, float* bias,
        float* output, int IN, int IC, int IH, int IW, int FH, int FW,
        int OC, int OH, int OW)
{

	float* src_input;
	int M = OC;
	int N = OH * OW;
	int K = IC * FH * FW;
	float alpha = 1.0;
	float beta = 0.0;

	src_input = (float*)malloc(sizeof(float) * IC * FH * FW * OH * OW * 4);
	struct timeval start, stop;
	gettimeofday(&start, NULL);
    for (int times = 0; times < LOOP_TIMES; ++ times) {

        rep(i, IN) {
            float* dst_output = output + i * OC * OH * OW;
            img2col(input + i * IC * IH * IW, src_input, OC, OH, OW, IC, IH, IW, FH, FW, true);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, filter, K, src_input, N, beta, dst_output, N);
        }

    }
	gettimeofday(&stop, NULL);
	float elapsed_time = (stop.tv_sec - start.tv_sec) +
        (float(stop.tv_usec - start.tv_usec) / 1e6);

	double gflops = ((2 * double(LOOP_TIMES) * double(IN) * double(IC) *
                double(OC) * double(OH) * double(OW) * double(FH) *
                double(FW)) / 1e9) / elapsed_time;

    float each_iter_time = elapsed_time / LOOP_TIMES;

    printf(" %f, %lf,", each_iter_time, gflops);

    for (int times = 0; times < LOOP_TIMES; ++times) {

        rep(i, IN) {
            float* dst_output = output + i * OC * OH * OW;
            rep(j, OC){
                rep(k, OH * OW){
                    dst_output[j * OH * OW + k] += bias[j];
                }
            }
        }

    }

	free(src_input);	
	return;
}
