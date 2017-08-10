#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <memory.h>
#include <string.h>
#include <math.h>

void DataInit(float* ptr, int length)
{
    srand(7);
    for (int i = 0; i < length; ++i)
    {
        int rand_num = rand();
        float value = rand_num;
        ptr[i] = value / RAND_MAX;
    }
}

int PtrComparison(const float* ptr1, const float* ptr2, int len)
{
    int cnt = 0;
    for (int i = 0; i < len; ++i)
    {
        if (fabs(ptr1[i] - ptr2[i]) < 1e-1)
        {
            cnt++;
            //printf("%d ptr1: %f, ptr2: %f\n", i, ptr1[i], ptr2[i]);
        }
        else
        {
            //printf(" %d ptr1: %f vs. ptr2: %f\n", i, ptr1[i], ptr2[i]);
            return 0;
        }
    }
    return 1;
}

extern void Im2colMKL(float* src, float* weights, float* bias,
        float* dst, int IN, int IC, int IH, int IW, int FH, int FW,
        int OC, int OH, int OW);
extern void MKLDNN(float* src, float* weights, float* bias,
        float* dst, int IN, int IC, int IH, int IW, int FH, int FW,
        int OC, int OH, int OW);

int main(int argc, char** argv)
{
    if (argc < 8){
        printf("Usage of the main: ./main IN IC IH IW FH FW OC\n");
        return 1;
    }

    int IN = atoi(argv[1]);
    int IC = atoi(argv[2]);
    int IH = atoi(argv[3]);
    int IW = atoi(argv[4]);
    int FH = atoi(argv[5]);
    int FW = atoi(argv[6]);
    int OC = atoi(argv[7]);
    int OH = IH - FH + 1;
    int OW = IW - FW + 1;

    float* input;
    input = (float*)malloc(sizeof(float) * IN * IC * IH * IW);
    DataInit(input, IN * IC * IH * IW);

    float* filter;
    filter = (float*)malloc(sizeof(float) * OC * IC * FH * FW);
    DataInit(filter, OC * IC * FH * FW);

    float* bias;
    bias = (float*)malloc(sizeof(float) * OC);
    DataInit(bias, OC);

    float* im2col_output;
    im2col_output = (float*)malloc(sizeof(float) * IN * OC * OH * OW);
    memset(im2col_output, 0, sizeof(float) * IN * OC * OH * OW);

    float* mkldnn_output;
    mkldnn_output = (float*)malloc(sizeof(float) * IN * OC * OH * OW);
    memset(mkldnn_output, 0, sizeof(float) * IN * OC * OH * OW);

    printf("%d, %d, %d, %d, %d, %d, %d,", IN, IC, IH, IW, FH, FW, OC);

    Im2colMKL(input, filter, bias, im2col_output, IN, IC, IH, IW, FH, FW,
            OC, OH, OW);

    MKLDNN(input, filter, bias, mkldnn_output, IN, IC, IH, IW, FH, FW,
            OC, OH, OW);

    if (PtrComparison(im2col_output, mkldnn_output, IN * OC * OH * OW))
        printf(" Accept!,");
    else
        printf(" Wrong Answer!,");

    free(input);
    free(filter);
    free(bias);
    free(im2col_output);
    free(mkldnn_output);

    printf("\n");
    return 0;
}
