#include <stdint.h>
#include <math.h>
#include <string.h>

typedef uint16_t ggml_fp16_t;

static inline uint32_t fp32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } tmp = { .f = f };
    return tmp.u;
}

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t u;
        float f;
    } tmp = { .u = w };
    return tmp.f;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

#define QK4_0 32
#define QK8_0 32

typedef struct {
    ggml_fp16_t d;
    uint8_t qs[QK4_0 / 2];
} block_q4_0;

typedef struct {
    ggml_fp16_t d;
    int8_t qs[QK8_0];
} block_q8_0;

void quant_ref_quantize_row_q4_0(const float * x, block_q4_0 * y, int64_t k) {
    const int qk = QK4_0;
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        float max = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            const float av = fabsf(v);
            if (amax < av) {
                amax = av;
                max  = v;
            }
        }

        const float d  = max / -8.0f;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = (uint8_t)(((int8_t)(x0 + 8.5f)) < 15 ? (int8_t)(x0 + 8.5f) : 15);
            const uint8_t xi1 = (uint8_t)(((int8_t)(x1 + 8.5f)) < 15 ? (int8_t)(x1 + 8.5f) : 15);

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quant_ref_quantize_row_q8_0(const float * x, block_q8_0 * y, int64_t k) {
    const int qk = QK8_0;
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = fabsf(x[i*qk + j]);
            if (amax < v) {
                amax = v;
            }
        }

        const float d  = amax / 127.0f;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk; j++) {
            const float x0 = x[i*qk + j]*id;
            const float x1 = roundf(x0);
            const float x2 = x1 < -128.0f ? -128.0f : (x1 > 127.0f ? 127.0f : x1);

            y[i].qs[j] = (int8_t) x2;
        }
    }
}

void quant_ref_dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k) {
    const int qk = QK4_0;
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j]       = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

void quant_ref_dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k) {
    const int qk = QK8_0;
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

float quant_ref_vec_dot_q4_0_q8_0(const block_q4_0 * x, const block_q8_0 * y, int64_t n) {
    const int qk = QK8_0;
    const int nb = n / qk;
    float sumf = 0.0f;

    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += v0 * y[ib].qs[j];
            sumi1 += v1 * y[ib].qs[j + qk/2];
        }

        sumf += (float)(sumi0 + sumi1) * GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d);
    }

    return sumf;
}
