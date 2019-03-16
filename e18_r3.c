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
#include <omp.h>


#define pprintf_abort(str, ...) \
    do { \
        fprintf(stderr, "%s:%u (%s): ", __FILE__, __LINE__, __func__); \
        fprintf(stderr, str, ##__VA_ARGS__); \
        fprintf(stderr, ": %s\n", strerror(errno)); \
        exit(EXIT_FAILURE); \
    } while (0)


/*
 * https://github.com/torvalds/linux/blob/master/include/linux/kernel.h
 * https://github.com/torvalds/linux/blob/master/include/linux/compiler.h
 */
#define barrier() __asm__ __volatile__("": : :"memory")
#define barrier_data(ptr) __asm__ __volatile__("": :"r"(ptr) :"memory")
#define __aligned(x) __attribute__((__aligned__(x)))
#define __assume_aligned(a, ...) __attribute__((__assume_aligned__(a, ## __VA_ARGS__)))
#define __always_unused __attribute__((__unused__))
#define __maybe_unused __attribute__((__unused__))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define BUILD_BUG_ON_ZERO(e) (sizeof(struct { int:(-!!(e)); }))
#define __must_be_array(a) BUILD_BUG_ON_ZERO(__same_type((a), &(a)[0]))
#define __same_type(a, b) __builtin_types_compatible_p(typeof(a), typeof(b))
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]) + __must_be_array(arr))
#define PAGE_MASK   (~(PAGE_SIZE-1))
#define __ALIGN_KERNEL(x, a)        __ALIGN_KERNEL_MASK(x, (typeof(x))(a) - 1)
#define __ALIGN_KERNEL_MASK(x, mask)    (((x) + (mask)) & ~(mask))
#define ALIGN(x, a)         __ALIGN_KERNEL((x), (a))
#define PAGE_ALIGN(addr) ALIGN(addr, PAGE_SIZE)
#define offset_in_page(p) ((unsigned long)(p) & ~PAGE_MASK)
#define __ALIGN_MASK(x, mask)   __ALIGN_KERNEL_MASK((x), (mask))
#define PTR_ALIGN(p, a)     ((typeof(p))ALIGN((unsigned long)(p), (a)))
#define IS_ALIGNED(x, a) (((x) & ((typeof(x))(a) - 1)) == 0)
#define __cmp(x, y, op) ((x) op (y) ? (x) : (y))
#define min(x, y) __careful_cmp(x, y, <)
#define max(x, y) __careful_cmp(x, y, >)
#define min3(x, y, z) min((typeof(x))min(x, y), z)
#define max3(x, y, z) max((typeof(x))max(x, y), z)

#define __cmp_once(x, y, op) \
    ({ \
        typeof(x) xx = (x); \
        typeof(y) yy = (y); \
        __cmp(xx, yy, op); \
    })
#define __careful_cmp(x, y, op) __cmp_once(x, y, op)

#define unreachable() \
    do { \
        fprintf(stderr, "%s:%d (%s): Unreachable!\n", \
                __FILE__, __LINE__, __func__); \
        exit(EXIT_FAILURE); \
        __builtin_unreachable(); \
    } while (0)

#define DIV_ROUND_DN(n,d) ((n) / (d))
#define DIV_ROUND_UP(n,d) (((n) - 1) / (d) + 1)

#define ORDER 6
#define SUM1 (ORDER * (ORDER*ORDER - 1) / 2)
#define SUM3 (SUM1 * 3)
#define SUM6 (SUM1 * 6)

typedef uint8_t elem_t;
typedef elem_t series_t [ORDER];
typedef uint64_t mask_t;

#define MASK(x) (((mask_t) 1) << (x))
#define USED(used,x)  ((used) &  MASK(x))

#define SIGMA(ini,n) ((((ini) + (ini) + (n) - 1) * (n)) >> 1)


#define SUM SUM1
#define STA 0
#define END ORDER*ORDER
#define DEPTH 6

#if SUM < SIGMA(STA,DEPTH)
#error "SUM is too small"
#endif
#if SIGMA(END-DEPTH,DEPTH) < SUM
#error "SUM is too large"
#endif

static inline int get_sta_end_1(int depth, int upper, int upper_sum, int *endp)
{
    int sta, end;
    const int d = DEPTH - depth;

    if (upper < 0)
        upper = max(SUM - SIGMA(END-DEPTH+1, DEPTH-1) - 1, STA - 1);

    sta = max(upper + 1, SUM - upper_sum - SIGMA(END - (d - 1), d - 1));

    if (unlikely(!(upper < sta && sta < END - (DEPTH - depth - 1)))) {
        fprintf(stderr, "%s:%d (%s): Invalid sta=%d: depth=%d upper=%d upper_sum=%d\n",
                __FILE__, __LINE__, __func__, sta, depth, upper, upper_sum);
        exit(EXIT_FAILURE);
    }

    end = max(upper + 1, (((SUM - upper_sum) << 1) - d*d + d) / (d << 1) + 1);

    if (unlikely(!(sta < end && end <= END - (DEPTH - depth - 1)))) {
        fprintf(stderr, "%s:%d (%s): Invalid end=%d with sta=%d: depth=%d upper=%d upper_sum=%d\n",
                __FILE__, __LINE__, __func__, end, sta, depth, upper, upper_sum);
        exit(EXIT_FAILURE);
    }

    *endp = end;
    return sta;
}

static inline uint64_t count_r3(mask_t used, uint32_t *n_series_p)
{
    mask_t series_mask[32134];
    uint32_t n_series = 0;

#if 0
    for (elem_t a0 = 0; a0 < ORDER*ORDER - 5; a0 ++) {
        if (USED(used,a0)) continue;
        for (elem_t a1 = a0 + 1; a1 < ORDER*ORDER - 4; a1 ++) {
            if (USED(used,a1)) continue;
            for (elem_t a2 = a1 + 1; a2 < ORDER*ORDER - 3; a2 ++) {
                if (USED(used,a2)) continue;
                for (elem_t a3 = a2 + 1; a3 < ORDER*ORDER - 2; a3 ++) {
                    if (USED(used,a3)) continue;
                    for (elem_t a4 = a3 + 1; a4 < ORDER*ORDER - 1; a4 ++) {
                        if (USED(used,a4)) continue;

                        int sum = a0 + a1 + a2 + a3 + a4;
                        if (sum > SUM1)
                            break;
                        sum = SUM1 - sum;
                        if (sum <= a4)
                            continue;
                        if (USED(used,sum))
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
#else
    int a0_end;
    for (int a0 = get_sta_end_1(0, -1, 0, &a0_end); a0 < a0_end; a0 ++) {
        if (USED(used,a0)) continue;
        int a1_end;
        for (int a1 = get_sta_end_1(1, a0, a0, &a1_end); a1 < a1_end; a1 ++) {
            if (USED(used,a1)) continue;
            int a2_end;
            for (int a2 = get_sta_end_1(2, a1, a0+a1, &a2_end); a2 < a2_end; a2 ++) {
                if (USED(used,a2)) continue;
                int a3_end;
                for (int a3 = get_sta_end_1(3, a2, a0+a1+a2, &a3_end); a3 < a3_end; a3 ++) {
                    if (USED(used,a3)) continue;
                    int a4_end;
                    for (int a4 = get_sta_end_1(4, a3, a0+a1+a2+a3, &a4_end); a4 < a4_end; a4 ++) {
                        if (USED(used,a4)) continue;
                        int a5 = SUM1 - (a0+a1+a2+a3+a4);
#if 0
                        if (!(a4 < a5 && a5 < END)) {
                            fprintf(stderr, "abc\n");
                            exit(EXIT_FAILURE);
                        }
#endif
                        if (USED(used,a5)) continue;

                        series_mask[n_series++] =
                              MASK(a0)
                            | MASK(a1)
                            | MASK(a2)
                            | MASK(a3)
                            | MASK(a4)
                            | MASK(a5);
                    }
                }
            }
        }
    }

#endif

    //printf("n_series = %" PRId32 "\n", n_series);

    if (n_series == 32134) {
        fprintf(stderr, "warning: n_series = %" PRIu32 ", used = 0x%09lx\n", n_series, used);
    }

    if (n_series > 32134) {
        fprintf(stderr, "error: n_series = %" PRIu32 ", used = 0x%09lx\n", n_series, used);
        exit(EXIT_FAILURE);
    }

    *n_series_p = n_series;

    uint64_t r3_count = 0;

    for (uint32_t i = 0; i < n_series - 2; i ++) {
        mask_t mi = series_mask[i];
        for (uint32_t j = i + 1; j < n_series - 1; j ++) {
            mask_t mj = series_mask[j];
            if (mi & mj)
                continue;
            mask_t mij = mi | mj;
            for (uint32_t k = j + 1; k < n_series - 0; k ++) {
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
#undef SUM
#undef STA
#undef END
#undef DEPTH

#define SUM SUM3

/* [STA, END) */

#ifndef STA
#define STA 0
#endif

#ifndef END
#define END (ORDER*ORDER)
#endif

#define DEPTH (ORDER*3)

#if SUM < SIGMA(STA,DEPTH)
#error "SUM is too small"
#endif
#if SIGMA(END-DEPTH,DEPTH) < SUM
#error "SUM is too large"
#endif

static double __maybe_unused getsec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double) t.tv_sec + t.tv_nsec * 1e-9;
}

static void *myvcalloc(size_t size)
{
    void *p;
    if (posix_memalign(&p, 4096, size)) {
        fprintf(stderr, "error: posix_memalign: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    (void) memset(p, 0, size);
    return p;
}

static inline int get_sta_end(int depth, int upper, int upper_sum, int *endp)
{
    int sta, end;
    const int d = DEPTH - depth;

    if (upper < 0)
        upper = max(SUM - SIGMA(END-DEPTH+1, DEPTH-1) - 1, STA - 1);

#if 0
    for (int i = upper + 1; i < END - (DEPTH - depth - 1); i ++) {
        if (upper_sum + i + SIGMA(END - (DEPTH - depth - 1), DEPTH - depth - 1) >= SUM) {
            sta = i;
            goto endloop1;
        }
    }
    fprintf(stderr, "%s:%d (%s): Unreachable: depth=%d upper=%d upper_sum=%d\n",
            __FILE__, __LINE__, __func__, depth, upper, upper_sum);
    exit(EXIT_FAILURE);

endloop1:
#else
    sta = max(upper + 1, SUM - upper_sum - SIGMA(END - (d - 1), d - 1));
#endif

    if (unlikely(!(upper < sta && sta < END - (DEPTH - depth - 1)))) {
        fprintf(stderr, "%s:%d (%s): Invalid sta=%d: depth=%d upper=%d upper_sum=%d\n",
                __FILE__, __LINE__, __func__, sta, depth, upper, upper_sum);
        exit(EXIT_FAILURE);
    }

#if 0
    for (int i = upper + 1; i < END - (DEPTH - depth - 1) + 1; i ++) {
        if (upper_sum + i + SIGMA(i + 1, DEPTH - depth - 1) > SUM) {
            end = i;
            goto endloop2;
        }
    }
    fprintf(stderr, "%s:%d (%s): Unreachable: depth=%d upper=%d upper_sum=%d\n",
            __FILE__, __LINE__, __func__, depth, upper, upper_sum);
    exit(EXIT_FAILURE);

endloop2:
#else
    /* I'm not so confident for this... */
    //end = max(upper + 1, ((((SUM - upper_sum) << 1) / (DEPTH - depth) - (DEPTH - depth - 1)) >> 1) + 1);
    end = max(upper + 1, (((SUM - upper_sum) << 1) - d*d + d) / (d << 1) + 1);
#endif

    if (unlikely(!(sta < end && end <= END - (DEPTH - depth - 1)))) {
        fprintf(stderr, "%s:%d (%s): Invalid end=%d with sta=%d: depth=%d upper=%d upper_sum=%d\n",
                __FILE__, __LINE__, __func__, end, sta, depth, upper, upper_sum);
        exit(EXIT_FAILURE);
    }

    *endp = end;
    return sta;
}

int main(void)
{
    //mask_t E18[113093022];
    mask_t *E18;
    size_t E18_len = 0;

    E18 = myvcalloc(113093022 * sizeof(*E18));

//#pragma omp parallel reduction(+:E18_len) reduction(max:count_max) reduction(+:count_sum) reduction(+:num_nonzero)
    {
        int i0_end;
//#pragma omp for collapse(1) schedule(static,1)
        for (int i0 = get_sta_end(0, -1, 0, &i0_end); i0 < i0_end; i0 ++) {
            int i1_end;
            for (int i1 = get_sta_end(1, i0, i0, &i1_end); i1 < i1_end; i1 ++) {
                int i2_end;
                for (int i2 = get_sta_end(2, i1, i0+i1, &i2_end); i2 < i2_end; i2 ++) {
                    int i3_end;
                    for (int i3 = get_sta_end(3, i2, i0+i1+i2, &i3_end); i3 < i3_end; i3 ++) {
                        int i4_end;
                        for (int i4 = get_sta_end(4, i3, i0+i1+i2+i3, &i4_end); i4 < i4_end; i4 ++) {
                            int i5_end;
                            for (int i5 = get_sta_end(5, i4, i0+i1+i2+i3+i4, &i5_end); i5 < i5_end; i5 ++) {
                                int i6_end;
                                for (int i6 = get_sta_end(6, i5, i0+i1+i2+i3+i4+i5, &i6_end); i6 < i6_end; i6 ++) {
                                    int i7_end;
                                    for (int i7 = get_sta_end(7, i6, i0+i1+i2+i3+i4+i5+i6, &i7_end); i7 < i7_end; i7 ++) {
                                        int i8_end;
                                        for (int i8 = get_sta_end(8, i7, i0+i1+i2+i3+i4+i5+i6+i7, &i8_end); i8 < i8_end; i8 ++) {
                                            int i9_end;
                                            for (int i9 = get_sta_end(9, i8, i0+i1+i2+i3+i4+i5+i6+i7+i8, &i9_end); i9 < i9_end; i9 ++) {
                                                int i10_end;
                                                for (int i10 = get_sta_end(10, i9, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9, &i10_end); i10 < i10_end; i10 ++) {
                                                    int i11_end;
                                                    for (int i11 = get_sta_end(11, i10, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10, &i11_end); i11 < i11_end; i11 ++) {
                                                        int i12_end;
                                                        for (int i12 = get_sta_end(12, i11, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10+i11, &i12_end); i12 < i12_end; i12 ++) {
                                                            int i13_end;
                                                            for (int i13 = get_sta_end(13, i12, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10+i11+i12, &i13_end); i13 < i13_end; i13 ++) {
                                                                int i14_end;
                                                                for (int i14 = get_sta_end(14, i13, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10+i11+i12+i13, &i14_end); i14 < i14_end; i14 ++) {
                                                                    int i15_end;
                                                                    for (int i15 = get_sta_end(15, i14, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10+i11+i12+i13+i14, &i15_end); i15 < i15_end; i15 ++) {
                                                                        int i16_end;
                                                                        for (int i16 = get_sta_end(16, i15, i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10+i11+i12+i13+i14+i15, &i16_end); i16 < i16_end; i16 ++) {
                                                                            int i17 = SUM - (i0+i1+i2+i3+i4+i5+i6+i7+i8+i9+i10+i11+i12+i13+i14+i15+i16);
#if 0
                                                                            if (i17 <= i16 || END <= i17) {
                                                                                fprintf(stderr, "aaa\n");
                                                                                exit(EXIT_FAILURE);
                                                                            }
#endif

                                                                            mask_t mask =
                                                                                  MASK(i0)
                                                                                | MASK(i1)
                                                                                | MASK(i2)
                                                                                | MASK(i3)
                                                                                | MASK(i4)
                                                                                | MASK(i5)
                                                                                | MASK(i6)
                                                                                | MASK(i7)
                                                                                | MASK(i8)
                                                                                | MASK(i9)
                                                                                | MASK(i10)
                                                                                | MASK(i11)
                                                                                | MASK(i12)
                                                                                | MASK(i13)
                                                                                | MASK(i14)
                                                                                | MASK(i15)
                                                                                | MASK(i16)
                                                                                | MASK(i17);

                                                                            E18[E18_len++] = mask;
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

    assert(E18_len == 113093022);

#define BLOCKSIZE 128

    size_t E18_pos = 0;
    uint64_t count_min = UINT64_MAX, count_max = 0, count_sum = 0, count_zeros = 0;
    uint32_t n_series_min = UINT32_MAX, n_series_max = 0, n_series_sum = 0, n_series_zeros = 0;

#pragma omp parallel shared(E18,E18_pos) firstprivate(E18_len) reduction(min:count_min) reduction(max:count_max) reduction(+:count_sum) reduction(+:count_zeros) reduction(min:n_series_min) reduction(max:n_series_max) reduction(+:n_series_sum) reduction(+:n_series_zeros)
    for (; ; ) {
        size_t pos;

#pragma omp atomic capture
        {
            pos = E18_pos;
            E18_pos += BLOCKSIZE;
        }

        if (pos >= E18_len)
            break;

        const ssize_t blocksize = min((size_t) BLOCKSIZE, E18_len - pos);
        for (ssize_t i = 0; i < blocksize; i ++) {
            mask_t mask = E18[pos + i];
            uint32_t n_series;
            //double start = getsec();
            uint64_t count = count_r3(mask, &n_series);
            //double end = getsec();

            if (count < count_min)
                count_min = count;
            if (count > count_max)
                count_max = count;
            count_sum += count;
            if (count == 0) {
                printf("count is zero on used=0x%09lx\n", mask);
                count_zeros++;
            }

            if (n_series < n_series_min)
                n_series_min = n_series;
            if (n_series > n_series_max)
                n_series_max = n_series;
            n_series_sum += n_series;
            if (n_series == 0) {
                printf("n_series is zero on used=0x%09lx\n", mask);
                n_series_zeros++;
            }

#if 0
            printf("%3d/%3d: time = %f [s], count = %" PRIu64 "\n",
                    omp_get_thread_num(), omp_get_num_threads(), end - start, count);
#endif
        }
    }

    printf("count_min = %" PRIu64 "\n", count_min);
    printf("count_max = %" PRIu64 "\n", count_max);
    printf("count_sum = %" PRIu64 "\n", count_sum);
    printf("count_zeros = %" PRIu64 "\n", count_zeros);

    printf("n_series_min = %" PRIu32 "\n", n_series_min);
    printf("n_series_max = %" PRIu32 "\n", n_series_max);
    printf("n_series_sum = %" PRIu32 "\n", n_series_sum);
    printf("n_series_zeros = %" PRIu32 "\n", n_series_zeros);

    return 0;
}
