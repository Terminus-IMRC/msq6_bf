#ifndef MADD_H_INCLUDED
#define MADD_H_INCLUDED

#include <immintrin.h>

/*
 * b: 8-bit
 * w: 16-bit
 * d: 32-bit
 * q: 64-bit
 */

/*
 * maddbw(a, b)
 *
 * Vertically multiply each *unsigned* 8-bit integer from a with the
 * corresponding *signed* 8-bit integer from b, producing intermediate *signed*
 * 16-bit integers.  Horizontally add adjacent pairs of intermediate 16-bit
 * integers, and pack the saturated results in dst.
 */

/* SSSE3 */
#define _mm_maddbw _mm_maddubs_epi16
/* AVX2 */
#define _mm256_maddbw _mm256_maddubs_epi16
/* AVX512BW */
#define _mm512_maddbw _mm512_maddubs_epi16
#define _mm512_mask_maddbw _mm512_mask_maddubs_epi16
#define _mm512_maskz_maddbw _mm512_maskz_maddubs_epi16
/* AVX512BW + AVX512VL */
#define _mm_mask_maddbw _mm_mask_maddubs_epi16
#define _mm_maskz_maddbw _mm_maskz_maddubs_epi16
#define _mm256_mask_maddbw _mm256_mask_maddubs_epi16
#define _mm256_maskz_maddbw _mm256_maskz_maddubs_epi16

/*
 * maddwd(a, b)
 *
 * Multiply packed *signed* 16-bit integers in a and b, producing intermediate
 * *signed* 32-bit integers.  Horizontally add adjacent pairs of intermediate
 * 32-bit integers, and pack the results in dst.
 */

/* SSE2 */
#define _mm_maddwd _mm_madd_epi16
/* AVX2 */
#define _mm256_maddwd _mm256_madd_epi16
/* AVX512BW */
#define _mm512_maddwd _mm512_madd_epi16
#define _mm512_mask_maddwd _mm512_mask_madd_epi16
#define _mm512_maskz_maddwd _mm512_maskz_madd_epi16
/* AVX512BW + AVX512VL */
#define _mm_mask_maddwd _mm_mask_madd_epi16
#define _mm_maskz_maddwd _mm_maskz_madd_epi16
#define _mm256_mask_maddwd _mm256_mask_madd_epi16
#define _mm256_maskz_maddwd _mm256_maskz_madd_epi16

/*
 * intrin            insn     conformance lat(skx) th(skx) port(skx) lat(knl) th(knl) port(knl)
 *    _mm_srli_si128  psrldq  SSE2         1        1      p5        13       13      FP0
 *    _mm_srli_epi64  psrlq   SSE2         1        0.5    p01       13       13      FP0
 * _mm256_srli_si256 vpsrldq  AVX2         1        1      p5        11        8      FP0
 * _mm256_srli_epi64 vpsrlq   AVX2         1        0.5    p01       11        8      FP0
 * _mm512_srli_epi64 vpsrlq   AVX512F      1        ?      p0         2?       1?     FP0
 * _mm_mul_epi32      pmuldq  SSE4.1       5        0.5    p01        6        2      FP0
 * _mm_mul_epu32      pmuludq SSE2         5        0.5    p01        6        0.5    FP0/1
 * _mm256_mul_epi32  vpmuldq  AVX2         5        0.5    p01        6        0.5    FP0/1
 * _mm256_mul_epu32  vpmuludq AVX2         5        0.5    p01        6        0.5    FP0/1
 * _mm512_mul_epi32  vpmuldq  AVX512F     10        2      2p0        6        0.5    FP0/1
 * _mm512_mul_epu32  vpmuludq AVX512F      5        1      p0         6        0.5    FP0/1
 * _mm_add_epi64      paddq   SSE2         1        0.33   p015       2        0.5    FP0/1
 * _mm256_add_epi64  vpaddq   AVX2         1        0.33   p015       2        0.5    FP0/1
 * _mm512_add_epi64  vpaddq   AVX512F      1        0.5    p05        2        0.5    FP0/1
 */

static inline
__m128i _mm_madddq(__m128i a, __m128i b)
{
    __m128i a2 = _mm_srli_epi64(a, 32);
    __m128i b2 = _mm_srli_epi64(b, 32);
    return _mm_add_epi64(
            _mm_mul_epu32(a, b),
            _mm_mul_epu32(a2, b2));
}

static inline
__m256i _mm256_madddq(__m256i a, __m256i b)
{
    __m256i a2 = _mm256_srli_epi64(a, 32);
    __m256i b2 = _mm256_srli_epi64(b, 32);
    return _mm256_add_epi64(
            _mm256_mul_epu32(a, b),
            _mm256_mul_epu32(a2, b2));
}

static inline
__m512i _mm512_madddq(__m512i a, __m512i b)
{
    __m512i a2 = _mm512_srli_epi64(a, 32);
    __m512i b2 = _mm512_srli_epi64(b, 32);
    return _mm512_add_epi64(
            _mm512_mul_epu32(a, b),
            _mm512_mul_epu32(a2, b2));
}

#endif /* MADD_H_INCLUDED */
