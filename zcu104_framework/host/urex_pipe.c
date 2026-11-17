/* urex_pipe.c -- PS-side streaming pipeline for the ZCU104 (ARM Cortex-A53) and for x86 testing.
 *
 *   raw event -> hot-pixel mask -> ANF (R=3 circle, 28 neighbours, <= 150 ms) -> 16x16 patch builder
 *             -> sorted list of (bin << 9 | addr) words, ready to push to the SNN core
 *
 * Bit-exact with python/pipeline.py (mask before ANF, ANF timestamp map updated by every event that
 * passes the mask, patch window t >= t_i - W over retained events with index <= i).
 * Also contains a software INT8 SNN (same arithmetic as rtl/snn_core.v) for CPU-only baselines.
 *
 * Build:  gcc -O2 -shared -fPIC -o liburex_pipe.so urex_pipe.c
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SW 1280
#define SH 720
#define CELL 8
#define CW (SW / CELL)
#define CH (SH / CELL)
#define NEVER INT64_MIN

typedef struct { int64_t t; uint16_t x, y; uint8_t p; } rev_t;
typedef struct { rev_t *e; int head, n, cap; } cell_t;

typedef struct {
    uint8_t  *hot;          /* SW*SH, 1 = masked */
    int64_t  *ts;           /* ANF last-timestamp map */
    int       anf_enabled;
    int64_t   anf_thr;
    int       R, T;
    int64_t   W;
    cell_t   *cells;
    uint8_t  *vox;          /* T*2*R*R scratch */
    int      *vlist;
    int       nanf;
    int       anf_dx[28], anf_dy[28];
} urex_ctx;

urex_ctx *urex_create(const int *hot_xy, int n_hot, int anf_enabled, int64_t anf_thr_us,
                      int R, int64_t window_us, int T) {
    urex_ctx *c = calloc(1, sizeof(urex_ctx));
    c->hot = calloc(SW * SH, 1);
    for (int i = 0; i < n_hot; i++) c->hot[hot_xy[2 * i] * SH + hot_xy[2 * i + 1]] = 1;
    c->ts = malloc(sizeof(int64_t) * SW * SH);
    for (int i = 0; i < SW * SH; i++) c->ts[i] = NEVER;
    c->anf_enabled = anf_enabled; c->anf_thr = anf_thr_us;
    c->R = R; c->W = window_us; c->T = T;
    c->cells = calloc(CW * CH, sizeof(cell_t));
    c->vox = calloc((size_t)T * 2 * R * R, 1);
    c->vlist = malloc(sizeof(int) * T * 2 * R * R);
    c->nanf = 0;
    for (int dx = -3; dx <= 3; dx++)
        for (int dy = -3; dy <= 3; dy++)
            if (dx * dx + dy * dy > 0 && dx * dx + dy * dy <= 9) { c->anf_dx[c->nanf] = dx; c->anf_dy[c->nanf] = dy; c->nanf++; }
    return c;
}

void urex_destroy(urex_ctx *c) {
    for (int i = 0; i < CW * CH; i++) free(c->cells[i].e);
    free(c->cells); free(c->hot); free(c->ts); free(c->vox); free(c->vlist); free(c);
}

static int anf_check(urex_ctx *c, int x, int y, int64_t t) {
    int keep = 0;
    for (int k = 0; k < c->nanf; k++) {
        int nx = x + c->anf_dx[k], ny = y + c->anf_dy[k];
        if (nx < 0 || nx >= SW || ny < 0 || ny >= SH) continue;
        int64_t tn = c->ts[nx * SH + ny];
        if (tn != NEVER && t - tn <= c->anf_thr) { keep = 1; break; }
    }
    c->ts[x * SH + y] = t;
    return keep;
}

static void cell_push(cell_t *cl, rev_t e) {
    if (cl->head + cl->n == cl->cap) {
        if (cl->head > 0) { memmove(cl->e, cl->e + cl->head, sizeof(rev_t) * cl->n); cl->head = 0; }
        if (cl->n == cl->cap) { cl->cap = cl->cap ? 2 * cl->cap : 16; cl->e = realloc(cl->e, sizeof(rev_t) * cl->cap); }
    }
    cl->e[cl->head + cl->n] = e; cl->n++;
}

static int cmp_int(const void *a, const void *b) { int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }

/* builds the patch for the retained event just inserted; returns number of (bin<<9|addr) words */
static int build_patch(urex_ctx *c, int x, int y, int64_t t, uint16_t *out) {
    int R = c->R, h = R / 2, T = c->T, nv = 0;
    int64_t t0 = t - c->W;
    int cx0 = (x - h) / CELL, cx1 = (x - h + R - 1) / CELL, cy0 = (y - h) / CELL, cy1 = (y - h + R - 1) / CELL;
    if (x - h < 0) cx0 = 0;
    if (y - h < 0) cy0 = 0;
    if (cx1 >= CW) cx1 = CW - 1;
    if (cy1 >= CH) cy1 = CH - 1;
    for (int cx = cx0; cx <= cx1; cx++)
        for (int cy = cy0; cy <= cy1; cy++) {
            cell_t *cl = &c->cells[cx * CH + cy];
            while (cl->n && cl->e[cl->head].t < t0) { cl->head++; cl->n--; }   /* prune: stream is time-ordered */
            for (int i = 0; i < cl->n; i++) {
                rev_t *e = &cl->e[cl->head + i];
                int xs = e->x - x + h, ys = e->y - y + h;
                if (xs < 0 || xs >= R || ys < 0 || ys >= R) continue;
                int64_t b = (e->t - t0) * T / (c->W + 1);
                if (b < 0) b = 0;
                if (b > T - 1) b = T - 1;
                int v = (int)b * 2 * R * R + e->p * R * R + ys * R + xs;
                if (!c->vox[v]) { c->vox[v] = 1; c->vlist[nv++] = v; }
            }
        }
    qsort(c->vlist, nv, sizeof(int), cmp_int);
    for (int i = 0; i < nv; i++) {
        int v = c->vlist[i]; c->vox[v] = 0;
        int b = v / (2 * R * R), a = v % (2 * R * R);
        out[i] = (uint16_t)((b << 9) | a);
    }
    return nv;
}

/* Process one event. Returns -1 masked, 0 rejected by ANF, 1 retained (patch written to out, length *n_out). */
int urex_push(urex_ctx *c, int x, int y, int p, int64_t t, uint16_t *out, int *n_out) {
    *n_out = 0;
    if (c->hot[x * SH + y]) return -1;
    if (c->anf_enabled && !anf_check(c, x, y, t)) return 0;
    rev_t e = { t, (uint16_t)x, (uint16_t)y, (uint8_t)p };
    cell_push(&c->cells[(x / CELL) * CH + (y / CELL)], e);
    *n_out = build_patch(c, x, y, t, out);
    return 1;
}

/* Batch version for Python: status[i] in {-1,0,1}; patches concatenated in ev_buf, offsets in ev_off (n+1).
 * Returns number of words written, or -1 if ev_cap is too small. */
int64_t urex_run(urex_ctx *c, int64_t n, const uint16_t *x, const uint16_t *y, const uint8_t *p, const int64_t *t,
                 int8_t *status, uint16_t *ev_buf, int64_t *ev_off, int64_t ev_cap) {
    int64_t pos = 0; int nv;
    uint16_t *tmp = malloc(sizeof(uint16_t) * c->T * 2 * c->R * c->R);
    for (int64_t i = 0; i < n; i++) {
        ev_off[i] = pos;
        status[i] = (int8_t)urex_push(c, x[i], y[i], p[i], t[i], tmp, &nv);
        if (pos + nv > ev_cap) { free(tmp); return -1; }
        memcpy(ev_buf + pos, tmp, sizeof(uint16_t) * nv); pos += nv;
    }
    ev_off[n] = pos;
    free(tmp);
    return pos;
}

/* ---------------- software INT8 SNN (same arithmetic as rtl/snn_core.v) ----------------
 * W: NIN*NH int8 row-major (W[addr*NH + j]); B, TH: int16; WD: int8. events sorted by bin. */
int32_t urex_snn(const int8_t *W, const int16_t *B, const int16_t *TH, const int8_t *WD,
                 int NIN, int NH, int T, int leak_num, int leak_shift,
                 const uint16_t *ev, int n_ev) {
    int32_t m[1024]; int32_t acc = 0; int i = 0;
    (void)NIN;
    for (int j = 0; j < NH; j++) m[j] = 0;
    for (int k = 0; k < T; k++) {
        for (int j = 0; j < NH; j++) m[j] = (int32_t)(((int64_t)m[j] * leak_num) >> leak_shift) + B[j];
        while (i < n_ev && (ev[i] >> 9) <= k) {
            const int8_t *row = W + (size_t)(ev[i] & 0x1FF) * NH;
            for (int j = 0; j < NH; j++) m[j] += row[j];
            i++;
        }
        for (int j = 0; j < NH; j++)
            if (m[j] > TH[j]) { m[j] -= TH[j]; acc += WD[j]; }
    }
    return acc;
}
