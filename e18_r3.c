#include <immintrin.h>
#include <sys/types.h>
#include <stdint.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <inttypes.h>

#define ORDER 6
#define SUM (ORDER * (ORDER*ORDER - 1) / 2)
/* 100 */
#define OFFSET ((35ULL+34+33) - (2+1+0) + 1)

typedef uint8_t elem_t;
typedef elem_t series_t [ORDER];
typedef uint64_t mask_t;

#define USE(x)   used |= ((mask_t) 1) << (x)
#define UNUSE(x) used ^= ((mask_t) 1) << (x)
#define USED(x)  used &  ((mask_t) 1) << (x)

static uint64_t count_r3(mask_t used)
{
    mask_t series_mask[32134];
    int32_t n_series = 0;

    for (elem_t a0 = 0; a0 < ORDER*ORDER - 5; a0 ++) {
        if (USED(a0)) continue;
        for (elem_t a1 = a0 + 1; a1 < ORDER*ORDER - 4; a1 ++) {
            if (USED(a1)) continue;
            for (elem_t a2 = a1 + 1; a2 < ORDER*ORDER - 3; a2 ++) {
                if (USED(a2)) continue;
                for (elem_t a3 = a2 + 1; a3 < ORDER*ORDER - 2; a3 ++) {
                    if (USED(a3)) continue;
                    for (elem_t a4 = a3 + 1; a4 < ORDER*ORDER - 1; a4 ++) {
                        if (USED(a4)) continue;

                        uint8_t sum = a0 + a1 + a2 + a3 + a4;
                        if (sum > SUM)
                            break;
                        sum = SUM - sum;
                        if (sum <= a4)
                            continue;
                        if (USED(sum))
                            continue;

                        series_mask[n_series++] =
                              (((mask_t) 1) << a0)
                            | (((mask_t) 1) << a1)
                            | (((mask_t) 1) << a2)
                            | (((mask_t) 1) << a3)
                            | (((mask_t) 1) << a4)
                            | (((mask_t) 1) << sum);
                    }
                }
            }
        }
    }

    //printf("n_series = %" PRId32 "\n", n_series);

    uint64_t r3_count = 0;

    for (int32_t i = 0; i < n_series - 2; i ++) {
        mask_t mi = series_mask[i];
        for (int32_t j = 1; j < n_series - 1; j ++) {
            mask_t mj = series_mask[j];
            if (mi & mj)
                continue;
            mask_t mij = mi | mj;
            for (int32_t k = 2; k < n_series - 0; k ++) {
                mask_t mk = series_mask[k];
                if (mij & mk)
                    continue;
                r3_count++;
            }
        }
    }

    //printf("r3_count = %" PRIu64 "\n", r3_count);
    return r3_count;
}

int main(void)
{
    size_t E18_len = 0;
    uint64_t count_max = 0, count_sum = 0, num_nonzero = 0;

#pragma omp parallel for collapse(6) schedule(dynamic,64) reduction(+:E18_len) reduction(max:count_max) reduction(+:count_sum) reduction(+:num_nonzero)
    for (uint8_t i0 = 0; i0 < ORDER*ORDER - 17; i0 ++) {
        for (uint8_t i1 = 1; i1 < ORDER*ORDER - 16; i1 ++) {
            for (uint8_t i2 = 2; i2 < ORDER*ORDER - 15; i2 ++) {
                for (uint8_t i3 = 3; i3 < ORDER*ORDER - 14; i3 ++) {
                    for (uint8_t i4 = 4; i4 < ORDER*ORDER - 13; i4 ++) {
                        for (uint8_t i5 = 5; i5 < ORDER*ORDER - 12; i5 ++) {
                            if (i0 >= i1 || i1 >= i2 || i2 >= i3 || i3 >= i4 || i4 >= i5)
                                continue;
                            for (uint8_t i6 = i5 + 1; i6 < ORDER*ORDER - 11; i6 ++) {
                                for (uint8_t i7 = i6 + 1; i7 < ORDER*ORDER - 10; i7 ++) {
                                    for (uint8_t i8 = i7 + 1; i8 < ORDER*ORDER - 9; i8 ++) {
                                        for (uint8_t i9 = i8 + 1; i9 < ORDER*ORDER - 8; i9 ++) {
                                            for (uint8_t i10 = i9 + 1; i10 < ORDER*ORDER - 7; i10 ++) {
                                                for (uint8_t i11 = i10 + 1; i11 < ORDER*ORDER - 6; i11 ++) {
                                                    for (uint8_t i12 = i11 + 1; i12 < ORDER*ORDER - 5; i12 ++) {
                                                        for (uint8_t i13 = i12 + 1; i13 < ORDER*ORDER - 4; i13 ++) {
                                                            for (uint8_t i14 = i13 + 1; i14 < ORDER*ORDER - 3; i14 ++) {
                                                                for (uint8_t i15 = i14 + 1; i15 < ORDER*ORDER - 2; i15 ++) {
                                                                    for (uint8_t i16 = i15 + 1; i16 < ORDER*ORDER - 1; i16 ++) {
                                                                        for (uint8_t i17 = i16 + 1; i17 < ORDER*ORDER - 0; i17 ++) {
                                                                            E18_len++;

                                                                            mask_t mask =
                                                                                  (((mask_t) 1) << i0)
                                                                                | (((mask_t) 1) << i1)
                                                                                | (((mask_t) 1) << i2)
                                                                                | (((mask_t) 1) << i3)
                                                                                | (((mask_t) 1) << i4)
                                                                                | (((mask_t) 1) << i5)
                                                                                | (((mask_t) 1) << i6)
                                                                                | (((mask_t) 1) << i7)
                                                                                | (((mask_t) 1) << i8)
                                                                                | (((mask_t) 1) << i9)
                                                                                | (((mask_t) 1) << i10)
                                                                                | (((mask_t) 1) << i11)
                                                                                | (((mask_t) 1) << i12)
                                                                                | (((mask_t) 1) << i13)
                                                                                | (((mask_t) 1) << i14)
                                                                                | (((mask_t) 1) << i15)
                                                                                | (((mask_t) 1) << i16)
                                                                                | (((mask_t) 1) << i17);

                                                                            uint64_t count = count_r3(mask);
                                                                            if (count > 0) {
                                                                                num_nonzero++;
                                                                                if (count > count_max)
                                                                                    count_max = count;
                                                                                count_sum += count;
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    printf("E18_len = %zu\n", E18_len);
    printf("count_max = %" PRIu64 "\n", count_max);
    printf("count_sum = %" PRIu64 "\n", count_sum);
    printf("num_nonzero = %" PRIu64 "\n", num_nonzero);

    return 0;
}
