/* membw.c — STREAM-triad memory bandwidth probe.
 *
 * Why this is in a model benchmark: LLM decode is memory-bandwidth-bound.
 * Each generated token streams the whole weight file through the CPU.
 * So tok/s ~= bandwidth / model_bytes. Measuring bandwidth turns a pile of
 * per-machine tok/s numbers into something you can actually explain and
 * extrapolate from.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N    (32L * 1024 * 1024)   /* 32M doubles = 256 MB per array */
#define REPS 5

int main(void) {
    double *a = aligned_alloc(64, N * sizeof(double));
    double *b = aligned_alloc(64, N * sizeof(double));
    double *c = aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "alloc failed\n"); return 1; }

    #pragma omp parallel for
    for (long i = 0; i < N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 3.0; }

    double best = 0.0;
    for (int r = 0; r < REPS; r++) {
        double t0 = omp_get_wtime();
        #pragma omp parallel for
        for (long i = 0; i < N; i++) a[i] = b[i] + 3.0 * c[i];
        double t1 = omp_get_wtime();
        double gb = 3.0 * sizeof(double) * N / 1e9;   /* STREAM convention */
        double bw = gb / (t1 - t0);
        if (bw > best) best = bw;
    }
    printf("%.1f\n", best);
    free(a); free(b); free(c);
    return 0;
}
