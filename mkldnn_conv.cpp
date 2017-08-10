#include <stdlib.h>
#include <stdio.h>
#include <mkldnn.h>
#include <sys/time.h>
#include <memory.h>
#include "param.h"

#define CHECK(f) do { \
    mkldnn_status_t s = f; \
    if (s != mkldnn_success) { \
        printf("[%s:%d] error: %s return %d,\n", __FILE__, __LINE__, #f, s); \
        exit(2); \
    } \
} while (0)

#define CHECK_TRUE(expr) do { \
    int e_ = expr; \
    if (!e_) { \
        printf("[%s:%d] %s failed!,\n", __FILE__, __LINE__, #expr); \
        exit(2); \
    } \
} while (0)

static void init_data_memory(uint32_t dim, const int *dims,
        mkldnn_memory_format_t user_fmt, mkldnn_data_type_t mkldnn_f32,
        mkldnn_engine_t engine, float *data, mkldnn_primitive_t *memory)
{
    mkldnn_memory_desc_t prim_md;
    mkldnn_primitive_desc_t user_pd;
    CHECK(mkldnn_memory_desc_init(&prim_md, dim, dims, mkldnn_f32, user_fmt));
    CHECK(mkldnn_memory_primitive_desc_create(&user_pd, &prim_md, engine));
    CHECK(mkldnn_primitive_create(memory, user_pd, NULL, NULL));

    void *req = NULL;
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == NULL);
    CHECK(mkldnn_memory_set_data_handle(*memory, data));
    CHECK(mkldnn_memory_get_data_handle(*memory, &req));
    CHECK_TRUE(req == data);
}

void MKLDNN(float* src, float* weights, float* bias, float* dst,
        int IN, int IC, int IH, int IW, int FH, int FW,
        int OC, int OH, int OW)
{
    mkldnn_engine_t engine;
    CHECK(mkldnn_engine_create(&engine, mkldnn_cpu, 0));

    int conv_src_sizes[4] = {IN, IC, IH, IW};
    int conv_weights_sizes[4] = {OC, IC, FH, FW};
    int conv_bias_sizes[4] = {OC};
    int conv_dst_sizes[4] = {IN, OC, OH, OW};
    int conv_strides[2] = {1, 1};
    int conv_padding[2] = {0, 0};

    mkldnn_primitive_t conv_user_src_memory, conv_user_weights_memory,
                       conv_user_bias_memory, conv_user_dst_memory;
    init_data_memory(4, conv_src_sizes, mkldnn_nchw, mkldnn_f32, engine,
            src, &conv_user_src_memory);
    init_data_memory(4, conv_weights_sizes, mkldnn_oihw, mkldnn_f32, engine,
            weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, mkldnn_x, mkldnn_f32, engine,
            bias, &conv_user_bias_memory);
    init_data_memory(4, conv_dst_sizes, mkldnn_nchw, mkldnn_f32, engine,
            dst, &conv_user_dst_memory);

    mkldnn_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
                         conv_dst_md;
    CHECK(mkldnn_memory_desc_init(&conv_src_md, 4, conv_src_sizes,
                mkldnn_f32, mkldnn_nchw));
    CHECK(mkldnn_memory_desc_init(&conv_weights_md, 4, conv_weights_sizes,
                mkldnn_f32, mkldnn_oihw));
    CHECK(mkldnn_memory_desc_init(&conv_bias_md, 1, conv_bias_sizes,
                mkldnn_f32, mkldnn_x));
    CHECK(mkldnn_memory_desc_init(&conv_dst_md, 4, conv_dst_sizes,
                mkldnn_f32, mkldnn_nchw));

    /* create a convolution */
    mkldnn_convolution_desc_t conv_any_desc;
    CHECK(mkldnn_convolution_forward_desc_init(&conv_any_desc, mkldnn_forward,
            mkldnn_convolution_direct, &conv_src_md, &conv_weights_md,
            &conv_bias_md, &conv_dst_md, conv_strides, conv_padding,
            conv_padding, mkldnn_padding_zero));

    mkldnn_primitive_desc_t conv_pd;
    CHECK(mkldnn_primitive_desc_create(&conv_pd, &conv_any_desc,
            engine, NULL));

    mkldnn_primitive_t conv_src_memory = conv_user_src_memory;
    mkldnn_primitive_t conv_weights_memory = conv_user_weights_memory;

    mkldnn_primitive_at_t conv_srcs[] = {
        mkldnn_primitive_at(conv_src_memory, 0),
        mkldnn_primitive_at(conv_weights_memory, 0),
        mkldnn_primitive_at(conv_user_bias_memory, 0)
    };

    const_mkldnn_primitive_t conv_dsts[] = { conv_user_dst_memory };

    /* finally create a convolution primitive */
    mkldnn_primitive_t conv;
    CHECK(mkldnn_primitive_create(&conv, conv_pd, conv_srcs, conv_dsts));

    uint32_t n = 0;
    mkldnn_primitive_t net[LOOP_TIMES * 2];

    for (int times; times < LOOP_TIMES; ++times){
        net[n++] = conv;
    }

    mkldnn_stream_t stream;
    CHECK(mkldnn_stream_create(&stream, mkldnn_eager));

    struct timeval start, stop;

    gettimeofday(&start, NULL);
    CHECK(mkldnn_stream_submit(stream, n, net, NULL));
    CHECK(mkldnn_stream_wait(stream, n, NULL));
    gettimeofday(&stop, NULL);

    float elapsed_time = (stop.tv_sec - start.tv_sec) +
        (float(stop.tv_usec - start.tv_usec) / 1e6);

	double gflops = ((2 * double(LOOP_TIMES) * double(IN) * double(IC) *
                double(OC) * double(OH) * double(OW) * double(FH) *
                double(FW)) / 1e9) / elapsed_time;

    float each_iter_time = elapsed_time / LOOP_TIMES;

    printf(" %f, %lf,", each_iter_time, gflops);

    /* clean-up */
    mkldnn_stream_destroy(stream);

    mkldnn_primitive_destroy(conv_user_src_memory);
    mkldnn_primitive_destroy(conv_user_weights_memory);
    mkldnn_primitive_destroy(conv_user_bias_memory);
    mkldnn_primitive_destroy(conv);

    return;
}
