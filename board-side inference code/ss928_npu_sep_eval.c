/*
 * npu_sep_eval.c  —  SS928 NPU speech-separation SI-SDR evaluator
 *
 * Usage:
 *   ./npu_sep_eval <model.om> <test_dir> [options]
 *
 * test_dir layout:
 *   test_dir/mix/  — mixed input PCM files
 *   test_dir/s1/   — reference speaker-1  (same filenames as mix/)
 *   test_dir/s2/   — reference speaker-2  (same filenames as mix/)
 *
 * Options:
 *   --pcm_dtype <s16|f32|s32>   PCM format  (default: s16)
 *   --max_files <N>             limit N files (default: 0 = all)
 *   --out_csv <path>            CSV output   (default: ./sisdr_result.csv)
 *   --device_id <id>            NPU device   (default: 0)
 *   --input_len <L>             override model input length (samples)
 *   --peak_norm <0|1>           normalize input by peak abs (default: 0)
 *   --out_layout <planar|interleaved|auto>
 *                               output layout decode mode (default: planar)
 *
 * Build:  see Makefile
 */

#include <acl.h>   /* SS928 SDK: mpp/out/include/npu/acl.h */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define EPS (1e-8)
#ifndef NAN
#define NAN (0.0 / 0.0)
#endif

/* ------------------------------------------------------------------ */
/*  Types                                                               */
/* ------------------------------------------------------------------ */
typedef enum {
    PCM_S16 = 0,
    PCM_F32 = 1,
    PCM_S32 = 2,
} pcm_type_t;

typedef enum {
    OUT_LAYOUT_PLANAR = 0,
    OUT_LAYOUT_INTERLEAVED = 1,
    OUT_LAYOUT_AUTO = 2,
} out_layout_t;

typedef struct {
    char **items;
    int   size;
    int   cap;
} str_vec_t;

typedef struct {
    char   *name;
    double  sisdr;
    double  sisdri;
    double  sdr;
    double  pesq;
    double  stoi;
    int     perm;       /* best permutation index (0 or 1) */
    double  infer_ms;
} row_t;

typedef struct {
    row_t *items;
    int    size;
    int    cap;
} row_vec_t;

typedef struct {
    char       model[PATH_MAX];
    char       test_dir[PATH_MAX];
    char       out_csv[PATH_MAX];
    char       dump_est_dir[PATH_MAX];
    char       pesq_cmd[PATH_MAX];
    char       stoi_cmd[PATH_MAX];
    char       tmp_dir[PATH_MAX];
    pcm_type_t pcm_type;
    out_layout_t out_layout;
    int        max_files;
    int        device_id;
    int        input_len_override;
    int        peak_norm;
    int        sample_rate;
    int        use_proxy_metrics;
    int        dump_est_pair; /* 0: save one aligned estimate, 1: save s1/s2 pair */
} args_t;

typedef struct {
    int            initialized;
    int            device_id;
    aclrtContext   context;
    uint32_t       model_id;
    aclmdlDesc    *model_desc;
    aclmdlDataset *input_ds;
    aclmdlDataset *output_ds;
    void          *input_dev;
    void          *output_dev;
    size_t         input_bytes;
    size_t         output_bytes;
    aclDataType    output_dtype;
    size_t         output_elem_size;
    int            input_len;      /* number of float samples */
    int            output_elems;
    int            output_channels;
    int            output_len_per_ch;
} npu_ctx_t;

/* ------------------------------------------------------------------ */
/*  str_vec helpers                                                     */
/* ------------------------------------------------------------------ */
static void str_vec_free(str_vec_t *v)
{
    int i;
    if (!v) return;
    for (i = 0; i < v->size; ++i) free(v->items[i]);
    free(v->items);
    v->items = NULL; v->size = 0; v->cap = 0;
}

static int str_vec_push(str_vec_t *v, const char *s)
{
    char **p; int nc;
    if (!v || !s) return -1;
    if (v->size == v->cap) {
        nc = v->cap ? v->cap * 2 : 16;
        p  = (char **)realloc(v->items, (size_t)nc * sizeof(char *));
        if (!p) return -1;
        v->items = p; v->cap = nc;
    }
    v->items[v->size] = strdup(s);
    if (!v->items[v->size]) return -1;
    v->size++;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  row_vec helpers                                                     */
/* ------------------------------------------------------------------ */
static void row_vec_free(row_vec_t *v)
{
    int i;
    if (!v) return;
    for (i = 0; i < v->size; ++i) free(v->items[i].name);
    free(v->items);
    v->items = NULL; v->size = 0; v->cap = 0;
}

static int row_vec_push(row_vec_t *v, const char *name,
                        double sisdr, double sisdri, double sdr,
                        double pesq, double stoi,
                        int perm, double infer_ms)
{
    row_t *p; int nc;
    if (!v || !name) return -1;
    if (v->size == v->cap) {
        nc = v->cap ? v->cap * 2 : 16;
        p  = (row_t *)realloc(v->items, (size_t)nc * sizeof(row_t));
        if (!p) return -1;
        v->items = p; v->cap = nc;
    }
    v->items[v->size].name     = strdup(name);
    v->items[v->size].sisdr    = sisdr;
    v->items[v->size].sisdri   = sisdri;
    v->items[v->size].sdr      = sdr;
    v->items[v->size].pesq     = pesq;
    v->items[v->size].stoi     = stoi;
    v->items[v->size].perm     = perm;
    v->items[v->size].infer_ms = infer_ms;
    if (!v->items[v->size].name) return -1;
    v->size++;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Path / file utilities                                               */
/* ------------------------------------------------------------------ */
static int cmp_cstr(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a, db = *(const double *)b;
    return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static bool file_exists(const char *p)
{
    struct stat st;
    return p && stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

static bool dir_exists(const char *p)
{
    struct stat st;
    return p && stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

static bool ends_with(const char *s, const char *suf)
{
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

static int join_path(char *out, size_t sz, const char *a, const char *b)
{
    size_t la, lb;
    if (!out || !a || !b || sz == 0) return -1;
    la = strlen(a);
    lb = strlen(b);
    if (la + 1 + lb + 1 > sz) {
        out[0] = '\0';
        return -1;
    }
    memcpy(out, a, la);
    out[la] = '/';
    memcpy(out + la + 1, b, lb + 1);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  PCM I/O                                                             */
/* ------------------------------------------------------------------ */
static int read_pcm_file(const char *path, pcm_type_t type,
                         float **out, int *out_len)
{
    FILE  *fp;
    long   bytes;
    size_t samples, i;
    float *buf = NULL;

    fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    bytes = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (bytes <= 0) { fclose(fp); return -1; }

    if (type == PCM_F32) {
        samples = (size_t)bytes / sizeof(float);
        buf = (float *)malloc(samples * sizeof(float));
        if (!buf || fread(buf, sizeof(float), samples, fp) != samples) {
            free(buf); fclose(fp); return -1;
        }
    } else if (type == PCM_S16) {
        int16_t *tmp;
        samples = (size_t)bytes / sizeof(int16_t);
        tmp = (int16_t *)malloc(samples * sizeof(int16_t));
        buf = (float *)malloc(samples * sizeof(float));
        if (!tmp || !buf ||
            fread(tmp, sizeof(int16_t), samples, fp) != samples) {
            free(tmp); free(buf); fclose(fp); return -1;
        }
        for (i = 0; i < samples; ++i)
            buf[i] = (float)((double)tmp[i] / 32768.0);
        free(tmp);
    } else {   /* PCM_S32 */
        int32_t *tmp;
        samples = (size_t)bytes / sizeof(int32_t);
        tmp = (int32_t *)malloc(samples * sizeof(int32_t));
        buf = (float *)malloc(samples * sizeof(float));
        if (!tmp || !buf ||
            fread(tmp, sizeof(int32_t), samples, fp) != samples) {
            free(tmp); free(buf); fclose(fp); return -1;
        }
        for (i = 0; i < samples; ++i)
            buf[i] = (float)((double)tmp[i] / 2147483648.0);
        free(tmp);
    }

    fclose(fp);
    *out = buf; *out_len = (int)samples;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  List .pcm files in a directory                                      */
/* ------------------------------------------------------------------ */
static int list_pcm_files(const char *dir_path, str_vec_t *out)
{
    DIR           *dir;
    struct dirent *ent;
    char           full[PATH_MAX];
    struct stat    st;

    dir = opendir(dir_path);
    if (!dir) return -1;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!ends_with(ent->d_name, ".pcm")) continue;
        if (join_path(full, sizeof(full), dir_path, ent->d_name)) continue;
        if (stat(full, &st) || !S_ISREG(st.st_mode)) continue;
        if (str_vec_push(out, ent->d_name)) { closedir(dir); return -1; }
    }
    closedir(dir);
    if (out->size > 1)
        qsort(out->items, (size_t)out->size, sizeof(char *), cmp_cstr);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  SI-SDR (zero-mean, scale-invariant)                                 */
/* ------------------------------------------------------------------ */
static double si_sdr(const float *est, const float *ref, int n)
{
    int    i;
    double em = 0.0, rm = 0.0, dot_er = 0.0, dot_rr = 0.0;
    double alpha, tp = 0.0, np = 0.0;

    if (!est || !ref || n <= 0) return -DBL_MAX;
    for (i = 0; i < n; ++i) { em += est[i]; rm += ref[i]; }
    em /= n; rm /= n;
    for (i = 0; i < n; ++i) {
        double e = est[i] - em, r = ref[i] - rm;
        dot_er += e * r; dot_rr += r * r;
    }
    alpha = dot_er / (dot_rr + EPS);
    for (i = 0; i < n; ++i) {
        double e = est[i] - em, r = ref[i] - rm;
        double s = alpha * r, ns = e - s;
        tp += s * s; np += ns * ns;
    }
    return 10.0 * log10((tp + EPS) / (np + EPS));
}

/*
 * PIT SI-SDR for 2-speaker separation.
 * est_multi: [ch0 | ch1] each 'n' floats (ch0 at offset 0, ch1 at offset n)
 * Returns mean SI-SDR of the best permutation; sets *best_perm.
 *   perm 0: ch0->s1, ch1->s2
 *   perm 1: ch0->s2, ch1->s1
 */
static double pit_sisdr_2ch(const float *ch0, const float *ch1,
                            const float *s1,  const float *s2,
                            int n, int *best_perm)
{
    double mean0 = (si_sdr(ch0, s1, n) + si_sdr(ch1, s2, n)) * 0.5;
    double mean1 = (si_sdr(ch0, s2, n) + si_sdr(ch1, s1, n)) * 0.5;
    if (mean0 >= mean1) { *best_perm = 0; return mean0; }
    *best_perm = 1; return mean1;
}

/* SDR (scale-dependent, no optimal rescaling). */
static double sdr_db(const float *est, const float *ref, int n)
{
    int i;
    double rp = 0.0, ep = 0.0;

    if (!est || !ref || n <= 0) return NAN;
    for (i = 0; i < n; ++i) {
        double r = ref[i];
        double e = est[i] - r;
        rp += r * r;
        ep += e * e;
    }
    return 10.0 * log10((rp + EPS) / (ep + EPS));
}

static int write_u16_le(FILE *fp, uint16_t v)
{
    unsigned char b[2];
    b[0] = (unsigned char)(v & 0xFFu);
    b[1] = (unsigned char)((v >> 8) & 0xFFu);
    return fwrite(b, 1, 2, fp) == 2 ? 0 : -1;
}

static int write_u32_le(FILE *fp, uint32_t v)
{
    unsigned char b[4];
    b[0] = (unsigned char)(v & 0xFFu);
    b[1] = (unsigned char)((v >> 8) & 0xFFu);
    b[2] = (unsigned char)((v >> 16) & 0xFFu);
    b[3] = (unsigned char)((v >> 24) & 0xFFu);
    return fwrite(b, 1, 4, fp) == 4 ? 0 : -1;
}

static int write_wav_mono_s16(const char *path, const float *x, int n, int sample_rate)
{
    FILE *fp;
    int i;
    uint32_t data_bytes;
    uint32_t riff_size;

    if (!path || !x || n <= 0 || sample_rate <= 0) return -1;
    if ((uint64_t)n > UINT32_MAX / 2u) return -1;
    data_bytes = (uint32_t)n * 2u;
    riff_size = 36u + data_bytes;

    fp = fopen(path, "wb");
    if (!fp) return -1;

    if (fwrite("RIFF", 1, 4, fp) != 4 ||
        write_u32_le(fp, riff_size) ||
        fwrite("WAVE", 1, 4, fp) != 4 ||
        fwrite("fmt ", 1, 4, fp) != 4 ||
        write_u32_le(fp, 16u) ||
        write_u16_le(fp, 1u) ||
        write_u16_le(fp, 1u) ||
        write_u32_le(fp, (uint32_t)sample_rate) ||
        write_u32_le(fp, (uint32_t)sample_rate * 2u) ||
        write_u16_le(fp, 2u) ||
        write_u16_le(fp, 16u) ||
        fwrite("data", 1, 4, fp) != 4 ||
        write_u32_le(fp, data_bytes)) {
        fclose(fp);
        return -1;
    }

    for (i = 0; i < n; ++i) {
        double v = x[i];
        int16_t s;
        if (v > 1.0) v = 1.0;
        if (v < -1.0) v = -1.0;
        s = (int16_t)lrint(v * 32767.0);
        if (write_u16_le(fp, (uint16_t)s)) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

static int write_pcm_mono_s16(const char *path, const float *x, int n)
{
    FILE *fp;
    int i;

    if (!path || !x || n <= 0) return -1;
    fp = fopen(path, "wb");
    if (!fp) return -1;

    for (i = 0; i < n; ++i) {
        double v = x[i];
        int16_t s;
        unsigned char b[2];
        if (v > 1.0) v = 1.0;
        if (v < -1.0) v = -1.0;
        s = (int16_t)lrint(v * 32767.0);
        b[0] = (unsigned char)((uint16_t)s & 0xFFu);
        b[1] = (unsigned char)(((uint16_t)s >> 8) & 0xFFu);
        if (fwrite(b, 1, 2, fp) != 2) {
            fclose(fp);
            return -1;
        }
    }

    fclose(fp);
    return 0;
}

static double parse_last_double(const char *s)
{
    const char *p = s;
    double last = NAN;
    while (p && *p) {
        char *end = NULL;
        double v;
        errno = 0;
        v = strtod(p, &end);
        if (end && end != p) {
            if (errno == 0 && isfinite(v)) last = v;
            p = end;
            continue;
        }
        ++p;
    }
    return last;
}

static double run_external_metric(const char *cmd,
                                  const char *tmp_dir,
                                  const char *metric_tag,
                                  const float *ref,
                                  const float *est,
                                  int n,
                                  int sample_rate,
                                  int file_idx,
                                  int pair_idx)
{
    char ref_wav[PATH_MAX], est_wav[PATH_MAX];
    char full_cmd[PATH_MAX * 3];
    char out_buf[4096];
    char line[512];
    size_t out_len = 0;
    FILE *pp = NULL;
    int pid = (int)getpid();
    double val;

    if (!cmd || !cmd[0] || !tmp_dir || !tmp_dir[0]) return NAN;
    if (!ref || !est || n <= 0 || sample_rate <= 0) return NAN;

    if (snprintf(ref_wav, sizeof(ref_wav), "%s/%s_ref_%d_%d_%d.wav",
                 tmp_dir, metric_tag, pid, file_idx, pair_idx) >= (int)sizeof(ref_wav) ||
        snprintf(est_wav, sizeof(est_wav), "%s/%s_est_%d_%d_%d.wav",
                 tmp_dir, metric_tag, pid, file_idx, pair_idx) >= (int)sizeof(est_wav)) {
        return NAN;
    }

    if (write_wav_mono_s16(ref_wav, ref, n, sample_rate) ||
        write_wav_mono_s16(est_wav, est, n, sample_rate)) {
        remove(ref_wav);
        remove(est_wav);
        return NAN;
    }

    if (snprintf(full_cmd, sizeof(full_cmd),
                 "%s \"%s\" \"%s\" %d",
                 cmd, ref_wav, est_wav, sample_rate) >= (int)sizeof(full_cmd)) {
        remove(ref_wav);
        remove(est_wav);
        return NAN;
    }

    pp = popen(full_cmd, "r");
    if (!pp) {
        remove(ref_wav);
        remove(est_wav);
        return NAN;
    }

    out_buf[0] = '\0';
    while (fgets(line, sizeof(line), pp) != NULL) {
        size_t l = strlen(line);
        if (out_len + l + 1 >= sizeof(out_buf)) break;
        memcpy(out_buf + out_len, line, l);
        out_len += l;
        out_buf[out_len] = '\0';
    }
    (void)pclose(pp);
    pp = NULL;

    remove(ref_wav);
    remove(est_wav);

    val = parse_last_double(out_buf);
    return val;
}

static double corrcoef_signal(const float *x, const float *y, int n)
{
    int i;
    double mx = 0.0, my = 0.0;
    double num = 0.0, dx2 = 0.0, dy2 = 0.0;

    if (!x || !y || n <= 1) return 0.0;
    for (i = 0; i < n; ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;

    for (i = 0; i < n; ++i) {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }

    return num / (sqrt(dx2 * dy2) + EPS);
}

/*
 * STOI proxy (no external dependency):
 * frame-wise correlation averaged on voiced frames, then mapped to [0,1].
 * This is NOT the official Taal STOI implementation.
 */
static double stoi_proxy_score(const float *ref, const float *est, int n)
{
    const int frame = 256;
    const int hop = 128;
    int i, s, valid = 0;
    double sum_corr = 0.0;

    if (!ref || !est || n <= 1) return NAN;
    if (n < frame) {
        double c = corrcoef_signal(ref, est, n);
        double s0 = 0.5 * (c + 1.0);
        if (s0 < 0.0) s0 = 0.0;
        if (s0 > 1.0) s0 = 1.0;
        return s0;
    }

    for (s = 0; s + frame <= n; s += hop) {
        double e = 0.0;
        for (i = 0; i < frame; ++i) {
            double r = ref[s + i];
            e += r * r;
        }
        if (e < 1e-6) continue; /* skip silence */
        sum_corr += corrcoef_signal(ref + s, est + s, frame);
        ++valid;
    }

    if (valid <= 0) {
        double c = corrcoef_signal(ref, est, n);
        double s0 = 0.5 * (c + 1.0);
        if (s0 < 0.0) s0 = 0.0;
        if (s0 > 1.0) s0 = 1.0;
        return s0;
    }

    {
        double c = sum_corr / valid;
        double s0 = 0.5 * (c + 1.0);
        if (s0 < 0.0) s0 = 0.0;
        if (s0 > 1.0) s0 = 1.0;
        return s0;
    }
}

static double seg_snr_proxy_db(const float *ref, const float *est, int n)
{
    const int frame = 512;
    const int hop = 256;
    int i, s, valid = 0;
    double sum = 0.0;

    if (!ref || !est || n <= 1) return NAN;
    if (n < frame) return sdr_db(est, ref, n);

    for (s = 0; s + frame <= n; s += hop) {
        double rp = 0.0, ep = 0.0, v;
        for (i = 0; i < frame; ++i) {
            double r = ref[s + i];
            double e = est[s + i] - r;
            rp += r * r;
            ep += e * e;
        }
        if (rp < 1e-8) continue;
        v = 10.0 * log10((rp + EPS) / (ep + EPS));
        if (v < -10.0) v = -10.0;
        if (v > 35.0) v = 35.0;
        sum += v;
        ++valid;
    }

    if (valid <= 0) return sdr_db(est, ref, n);
    return sum / valid;
}

/*
 * PESQ proxy (no external dependency):
 * combine segSNR + STOI proxy + SDR and map to [1.0, 4.5].
 * This is NOT ITU-T P.862 PESQ.
 */
static double pesq_proxy_score(const float *ref, const float *est, int n, int sample_rate)
{
    double stoi_v, ssnr_v, sdr_v, v;
    (void)sample_rate;
    if (!ref || !est || n <= 1) return NAN;
    stoi_v = stoi_proxy_score(ref, est, n);
    ssnr_v = seg_snr_proxy_db(ref, est, n);
    sdr_v = sdr_db(est, ref, n);

    v = 1.0 + 0.03 * (ssnr_v + 10.0) + 1.50 * stoi_v + 0.01 * (sdr_v + 5.0);
    if (v < 1.0) v = 1.0;
    if (v > 4.5) v = 4.5;
    return v;
}

static void compute_extra_metrics(const args_t *args,
                                  int file_idx,
                                  const float *mix,
                                  const float *est,
                                  int channels,
                                  int valid_len,
                                  int best_perm,
                                  const float *s1,
                                  const float *s2,
                                  int s2_len,
                                  bool use_s2,
                                  double sisdr_score,
                                  double *sisdri_out,
                                  double *sdr_out,
                                  double *pesq_out,
                                  double *stoi_out)
{
    if (sisdri_out) *sisdri_out = NAN;
    if (sdr_out) *sdr_out = NAN;
    if (pesq_out) *pesq_out = NAN;
    if (stoi_out) *stoi_out = NAN;
    if (!mix || !est || !s1 || valid_len <= 0 || channels <= 0) return;

    if (use_s2 && s2 && channels >= 2) {
        int n = valid_len < s2_len ? valid_len : s2_len;
        const float *ch0 = est;
        const float *ch1 = est + valid_len;
        const float *e1 = (best_perm == 1) ? ch1 : ch0;
        const float *e2 = (best_perm == 1) ? ch0 : ch1;
        double mix_sisdr = (si_sdr(mix, s1, n) + si_sdr(mix, s2, n)) * 0.5;
        double pesq1 = NAN, pesq2 = NAN, stoi1 = NAN, stoi2 = NAN;

        if (sisdri_out) *sisdri_out = sisdr_score - mix_sisdr;
        if (sdr_out) *sdr_out = (sdr_db(e1, s1, n) + sdr_db(e2, s2, n)) * 0.5;

        if (pesq_out) {
            if (args->pesq_cmd[0]) {
                pesq1 = run_external_metric(args->pesq_cmd, args->tmp_dir, "pesq",
                                            s1, e1, n, args->sample_rate, file_idx, 0);
                pesq2 = run_external_metric(args->pesq_cmd, args->tmp_dir, "pesq",
                                            s2, e2, n, args->sample_rate, file_idx, 1);
                if (isfinite(pesq1) && isfinite(pesq2)) *pesq_out = (pesq1 + pesq2) * 0.5;
            } else if (args->use_proxy_metrics) {
                *pesq_out = (pesq_proxy_score(s1, e1, n, args->sample_rate) +
                             pesq_proxy_score(s2, e2, n, args->sample_rate)) * 0.5;
            }
        }
        if (stoi_out) {
            if (args->stoi_cmd[0]) {
                stoi1 = run_external_metric(args->stoi_cmd, args->tmp_dir, "stoi",
                                            s1, e1, n, args->sample_rate, file_idx, 0);
                stoi2 = run_external_metric(args->stoi_cmd, args->tmp_dir, "stoi",
                                            s2, e2, n, args->sample_rate, file_idx, 1);
                if (isfinite(stoi1) && isfinite(stoi2)) *stoi_out = (stoi1 + stoi2) * 0.5;
            } else if (args->use_proxy_metrics) {
                *stoi_out = (stoi_proxy_score(s1, e1, n) +
                             stoi_proxy_score(s2, e2, n)) * 0.5;
            }
        }
    } else {
        int c = best_perm;
        const float *ebest;
        double mix_sisdr;
        if (c < 0 || c >= channels) c = 0;
        ebest = est + (size_t)c * (size_t)valid_len;
        mix_sisdr = si_sdr(mix, s1, valid_len);

        if (sisdri_out) *sisdri_out = sisdr_score - mix_sisdr;
        if (sdr_out) *sdr_out = sdr_db(ebest, s1, valid_len);
        if (pesq_out) {
            if (args->pesq_cmd[0]) {
                *pesq_out = run_external_metric(args->pesq_cmd, args->tmp_dir, "pesq",
                                                s1, ebest, valid_len, args->sample_rate, file_idx, 0);
            } else if (args->use_proxy_metrics) {
                *pesq_out = pesq_proxy_score(s1, ebest, valid_len, args->sample_rate);
            }
        }
        if (stoi_out) {
            if (args->stoi_cmd[0]) {
                *stoi_out = run_external_metric(args->stoi_cmd, args->tmp_dir, "stoi",
                                                s1, ebest, valid_len, args->sample_rate, file_idx, 0);
            } else if (args->use_proxy_metrics) {
                *stoi_out = stoi_proxy_score(s1, ebest, valid_len);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Timing helper                                                        */
/* ------------------------------------------------------------------ */
static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e3 + (double)tv.tv_usec * 1e-3;
}

/* ------------------------------------------------------------------ */
/*  Median                                                               */
/* ------------------------------------------------------------------ */
static double calc_median(const double *arr, int n)
{
    double *tmp, med;
    if (!arr || n <= 0) return 0.0;
    tmp = (double *)malloc((size_t)n * sizeof(double));
    if (!tmp) return 0.0;
    memcpy(tmp, arr, (size_t)n * sizeof(double));
    qsort(tmp, (size_t)n, sizeof(double), cmp_double);
    med = (n & 1) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);
    return med;
}

enum {
    MET_SISDR = 0,
    MET_SISDRI = 1,
    MET_SDR = 2,
    MET_PESQ = 3,
    MET_STOI = 4
};

static double row_metric_value(const row_t *r, int metric_id)
{
    if (!r) return NAN;
    switch (metric_id) {
        case MET_SISDR: return r->sisdr;
        case MET_SISDRI: return r->sisdri;
        case MET_SDR: return r->sdr;
        case MET_PESQ: return r->pesq;
        case MET_STOI: return r->stoi;
        default: return NAN;
    }
}

static int calc_metric_stats(const row_vec_t *rows, int metric_id,
                             int *finite_n,
                             double *mean,
                             double *median,
                             double *min_v,
                             double *max_v)
{
    double *vals;
    double sum = 0.0;
    int i, n = 0;

    if (!rows || rows->size <= 0 ||
        !finite_n || !mean || !median || !min_v || !max_v) return -1;

    vals = (double *)malloc((size_t)rows->size * sizeof(double));
    if (!vals) return -1;

    *min_v = DBL_MAX;
    *max_v = -DBL_MAX;

    for (i = 0; i < rows->size; ++i) {
        double v = row_metric_value(&rows->items[i], metric_id);
        if (!isfinite(v)) continue;
        vals[n++] = v;
        sum += v;
        if (v < *min_v) *min_v = v;
        if (v > *max_v) *max_v = v;
    }

    *finite_n = n;
    if (n <= 0) {
        *mean = *median = NAN;
        *min_v = *max_v = NAN;
        free(vals);
        return -1;
    }

    *mean = sum / n;
    *median = calc_median(vals, n);
    free(vals);
    return 0;
}

static const char *layout_to_str(out_layout_t layout)
{
    if (layout == OUT_LAYOUT_PLANAR) return "planar";
    if (layout == OUT_LAYOUT_INTERLEAVED) return "interleaved";
    return "auto";
}

/* ------------------------------------------------------------------ */
/*  Output decode helpers                                               */
/* ------------------------------------------------------------------ */
static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    float f;
    if (exp == 0) {
        f = (mant == 0) ? 0.0f : ldexpf((float)mant, -24);
    } else if (exp == 31) {
        f = (mant == 0) ? (1.0f / 0.0f) : (0.0f / 0.0f);
    } else {
        f = ldexpf((float)(mant | 0x400u), (int)exp - 25);
    }
    return sign ? -f : f;
}

static int decode_output_to_f32(const npu_ctx_t *ctx, const void *src, float *dst)
{
    int i;
    if (!ctx || !src || !dst || ctx->output_elems <= 0) return -1;
    if (ctx->output_dtype == ACL_FLOAT16) {
        const uint16_t *h = (const uint16_t *)src;
        for (i = 0; i < ctx->output_elems; ++i)
            dst[i] = fp16_to_fp32(h[i]);
        return 0;
    }
    if (ctx->output_dtype == ACL_FLOAT) {
        memcpy(dst, src, (size_t)ctx->output_elems * sizeof(float));
        return 0;
    }
    printf("[ERROR] unsupported output dtype=%d\n", (int)ctx->output_dtype);
    return -1;
}

/* ------------------------------------------------------------------ */
/*  NPU lifecycle                                                        */
/* ------------------------------------------------------------------ */
static int recalc_output_layout(npu_ctx_t *ctx)
{
    if (!ctx || ctx->input_len <= 0 || ctx->output_elems <= 0) return -1;
    if (ctx->output_elems % ctx->input_len == 0) {
        ctx->output_channels    = ctx->output_elems / ctx->input_len;
        ctx->output_len_per_ch  = ctx->input_len;
    } else {
        ctx->output_channels    = 1;
        ctx->output_len_per_ch  = ctx->output_elems;
    }
    if (ctx->output_channels   <= 0) ctx->output_channels   = 1;
    if (ctx->output_len_per_ch <= 0) ctx->output_len_per_ch = ctx->input_len;
    return 0;
}

static int npu_init(npu_ctx_t *ctx, const char *model_path, int device_id)
{
    aclError       ret;
    aclDataBuffer *in_buf, *out_buf;

    if (!ctx || !model_path) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->device_id = -1;

    ret = aclInit(NULL);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclInit failed, ret=%d\n", ret); return -1;
    }
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtSetDevice(%d) failed, ret=%d\n", device_id, ret);
        goto fail;
    }
    ctx->device_id = device_id;

    ret = aclrtCreateContext(&ctx->context, device_id);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtCreateContext failed, ret=%d\n", ret); goto fail;
    }

    ret = aclmdlLoadFromFile(model_path, &ctx->model_id);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclmdlLoadFromFile(%s) failed, ret=%d\n",
               model_path, ret);
        goto fail;
    }

    ctx->model_desc = aclmdlCreateDesc();
    if (!ctx->model_desc) {
        printf("[ERROR] aclmdlCreateDesc failed\n"); goto fail;
    }
    ret = aclmdlGetDesc(ctx->model_desc, ctx->model_id);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclmdlGetDesc failed, ret=%d\n", ret); goto fail;
    }

    ctx->input_bytes   = aclmdlGetInputSizeByIndex(ctx->model_desc,  0);
    ctx->output_bytes  = aclmdlGetOutputSizeByIndex(ctx->model_desc, 0);
    ctx->output_dtype  = aclmdlGetOutputDataType(ctx->model_desc, 0);
    if (ctx->output_dtype == ACL_FLOAT16) {
        ctx->output_elem_size = sizeof(uint16_t);
    } else if (ctx->output_dtype == ACL_FLOAT) {
        ctx->output_elem_size = sizeof(float);
    } else {
        printf("[ERROR] unsupported output dtype=%d\n", (int)ctx->output_dtype);
        goto fail;
    }
    ctx->input_len     = (int)(ctx->input_bytes  / sizeof(float));
    ctx->output_elems  = (int)(ctx->output_bytes / ctx->output_elem_size);

    if (ctx->input_len <= 0 || ctx->output_elems <= 0) {
        printf("[ERROR] invalid model I/O: input=%zu bytes, output=%zu bytes, out_dtype=%d, elem_size=%zu\n",
               ctx->input_bytes, ctx->output_bytes, (int)ctx->output_dtype, ctx->output_elem_size);
        goto fail;
    }
    printf("[INFO] model io: input_bytes=%zu output_bytes=%zu out_dtype=%d out_elem_size=%zu out_elems=%d\n",
           ctx->input_bytes, ctx->output_bytes, (int)ctx->output_dtype, ctx->output_elem_size, ctx->output_elems);
    if (recalc_output_layout(ctx)) goto fail;

    ret = aclrtMalloc(&ctx->input_dev,  ctx->input_bytes,
                      ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtMalloc input failed, ret=%d\n", ret); goto fail;
    }
    ret = aclrtMalloc(&ctx->output_dev, ctx->output_bytes,
                      ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclrtMalloc output failed, ret=%d\n", ret); goto fail;
    }

    ctx->input_ds  = aclmdlCreateDataset();
    ctx->output_ds = aclmdlCreateDataset();
    if (!ctx->input_ds || !ctx->output_ds) {
        printf("[ERROR] aclmdlCreateDataset failed\n"); goto fail;
    }

    in_buf  = aclCreateDataBuffer(ctx->input_dev,  ctx->input_bytes);
    out_buf = aclCreateDataBuffer(ctx->output_dev, ctx->output_bytes);
    if (!in_buf || !out_buf) {
        printf("[ERROR] aclCreateDataBuffer failed\n"); goto fail;
    }
    if (aclmdlAddDatasetBuffer(ctx->input_ds,  in_buf)  != ACL_ERROR_NONE ||
        aclmdlAddDatasetBuffer(ctx->output_ds, out_buf) != ACL_ERROR_NONE) {
        printf("[ERROR] aclmdlAddDatasetBuffer failed\n"); goto fail;
    }

    ctx->initialized = 1;
    return 0;
fail:
    return -1;
}

static void destroy_dataset(aclmdlDataset *ds)
{
    size_t i, n;
    if (!ds) return;
    n = aclmdlGetDatasetNumBuffers(ds);
    for (i = 0; i < n; ++i) {
        aclDataBuffer *b = aclmdlGetDatasetBuffer(ds, i);
        if (b) aclDestroyDataBuffer(b);
    }
    aclmdlDestroyDataset(ds);
}

static void npu_deinit(npu_ctx_t *ctx)
{
    if (!ctx) return;
    if (ctx->output_ds) { destroy_dataset(ctx->output_ds); ctx->output_ds = NULL; }
    if (ctx->input_ds)  { destroy_dataset(ctx->input_ds);  ctx->input_ds  = NULL; }
    if (ctx->output_dev){ aclrtFree(ctx->output_dev); ctx->output_dev = NULL; }
    if (ctx->input_dev) { aclrtFree(ctx->input_dev);  ctx->input_dev  = NULL; }
    if (ctx->model_desc){ aclmdlDestroyDesc(ctx->model_desc); ctx->model_desc = NULL; }
    if (ctx->model_id)  { aclmdlUnload(ctx->model_id); ctx->model_id = 0; }
    if (ctx->context)   { aclrtDestroyContext(ctx->context); ctx->context = NULL; }
    if (ctx->device_id >= 0) aclrtResetDevice(ctx->device_id);
    aclFinalize();
}

/* ------------------------------------------------------------------ */
/*  Single-chunk inference (input_len samples in, output_bytes out)     */
/* ------------------------------------------------------------------ */
static int npu_infer_chunk(const npu_ctx_t *ctx,
                           const float *in_host, void *out_host)
{
    aclError ret;
    ret = aclrtMemcpy(ctx->input_dev, ctx->input_bytes,
                      in_host, ctx->input_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] H2D memcpy failed, ret=%d\n", ret); return -1;
    }
    ret = aclmdlExecute(ctx->model_id, ctx->input_ds, ctx->output_ds);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] aclmdlExecute failed, ret=%d\n", ret); return -1;
    }
    ret = aclrtMemcpy(out_host, ctx->output_bytes,
                      ctx->output_dev, ctx->output_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        printf("[ERROR] D2H memcpy failed, ret=%d\n", ret); return -1;
    }
    return 0;
}

/*
 * Infer a waveform of arbitrary length.
 * Pads to a multiple of input_len, runs chunk-by-chunk, trims to n.
 * Returns interleaved buffer [ch0(n) | ch1(n) | ...] or NULL on error.
 * Caller must free *est_multi.
 */
static int infer_wave(const npu_ctx_t *ctx,
                      const float *x, int n,
                      int peak_norm,
                      out_layout_t layout,
                      float **est_multi, int *channels,
                      double *infer_ms_out)
{
    float   *xpad = NULL, *all = NULL, *trim = NULL, *chunk_f32 = NULL;
    uint8_t *chunk_raw = NULL;
    int    L, ch, pad, total, off, c, t, copy_n;
    double t0, t1;

    L  = ctx->input_len;
    ch = ctx->output_channels;
    if (L <= 0 || ch <= 0) return -1;
    if (layout == OUT_LAYOUT_AUTO) return -1;

    pad   = (L - (n % L)) % L;
    total = n + pad;

    xpad      = (float *)calloc((size_t)total, sizeof(float));
    chunk_raw = (uint8_t *)malloc(ctx->output_bytes);
    chunk_f32 = (float *)malloc((size_t)ctx->output_elems * sizeof(float));
    all       = (float *)calloc((size_t)ch * (size_t)total, sizeof(float));
    if (!xpad || !chunk_raw || !chunk_f32 || !all) goto fail;

    memcpy(xpad, x, (size_t)n * sizeof(float));
    if (peak_norm) {
        float peak = 0.0f;
        for (t = 0; t < n; ++t) {
            float v = fabsf(xpad[t]);
            if (v > peak) peak = v;
        }
        if (peak > 1e-12f) {
            float inv_peak = 1.0f / peak;
            for (t = 0; t < n; ++t) xpad[t] *= inv_peak;
        }
    }
    copy_n = ctx->output_len_per_ch < L ? ctx->output_len_per_ch : L;

    t0 = now_ms();
    for (off = 0; off < total; off += L) {
        if (npu_infer_chunk(ctx, xpad + off, chunk_raw)) goto fail;
        if (decode_output_to_f32(ctx, chunk_raw, chunk_f32)) goto fail;
        for (c = 0; c < ch; ++c) {
            float *dst = all + (size_t)c * total + off;
            if (layout == OUT_LAYOUT_PLANAR) {
                float *src = chunk_f32 + (size_t)c * ctx->output_len_per_ch;
                memcpy(dst, src, (size_t)copy_n * sizeof(float));
            } else { /* OUT_LAYOUT_INTERLEAVED */
                for (t = 0; t < copy_n; ++t)
                    dst[t] = chunk_f32[(size_t)t * ch + c];
            }
        }
    }
    t1 = now_ms();
    if (infer_ms_out) *infer_ms_out = t1 - t0;

    trim = (float *)malloc((size_t)ch * (size_t)n * sizeof(float));
    if (!trim) goto fail;
    for (c = 0; c < ch; ++c)
        memcpy(trim + (size_t)c * n, all + (size_t)c * total,
               (size_t)n * sizeof(float));

    free(xpad); free(chunk_raw); free(chunk_f32); free(all);
    *est_multi = trim; *channels = ch;
    return 0;
fail:
    free(xpad); free(chunk_raw); free(chunk_f32); free(all);
    return -1;
}

static double score_estimate(const float *est, int channels, int valid_len,
                             const float *s1,
                             const float *s2, int s2_len,
                             bool use_s2, int *best_perm)
{
    if (use_s2 && channels >= 2 && s2) {
        int n2 = valid_len < s2_len ? valid_len : s2_len;
        return pit_sisdr_2ch(est, est + valid_len, s1, s2, n2, best_perm);
    }

    {
        int c;
        double best = -DBL_MAX;
        for (c = 0; c < channels; ++c) {
            double v = si_sdr(est + (size_t)c * valid_len, s1, valid_len);
            if (v > best) { best = v; *best_perm = c; }
        }
        return best;
    }
}

/* ------------------------------------------------------------------ */
/*  Argument parsing                                                     */
/* ------------------------------------------------------------------ */
static void print_usage(const char *bin)
{
    printf("Usage:\n");
    printf("  %s <model.om> <test_dir> [options]\n\n", bin);
    printf("test_dir must contain: mix/  s1/  s2/  (*.pcm files)\n\n");
    printf("Options:\n");
    printf("  --pcm_dtype <s16|f32|s32>   PCM format     (default: s16)\n");
    printf("  --max_files <N>             limit files    (default: 0=all)\n");
    printf("  --out_csv   <path>          CSV output     (default: ./sisdr_result.csv)\n");
    printf("  --dump_est_dir <dir>        dump est_s1 PCM (estimate_xxxx.pcm)\n");
    printf("  --dump_est_pair <0|1>       when dumping, save aligned s1/s2 pair (estimate_xxxx_s1/s2.pcm)\n");
    printf("  --device_id <id>            NPU device id  (default: 0)\n");
    printf("  --input_len <L>             override model input length\n");
    printf("  --peak_norm <0|1>           input peak normalize (default: 0)\n");
    printf("  --out_layout <planar|interleaved|auto> (default: planar)\n");
    printf("  --sample_rate <Hz>          sample rate for PESQ/STOI (default: 16000)\n");
    printf("  --tmp_dir <dir>             temp dir for PESQ/STOI wavs (default: /tmp)\n");
    printf("  --pesq_cmd <cmd>            external PESQ cmd: <cmd> ref.wav est.wav sr\n");
    printf("  --stoi_cmd <cmd>            external STOI cmd: <cmd> ref.wav est.wav sr\n");
    printf("  --proxy_metrics <0|1>       use built-in proxy PESQ/STOI when cmd missing (default: 1)\n");
}

static int parse_args(int argc, char **argv, args_t *args)
{
    int i;
    if (!args) return -1;
    memset(args, 0, sizeof(*args));
    args->pcm_type  = PCM_S16;
    args->out_layout = OUT_LAYOUT_PLANAR;
    args->max_files = 0;
    args->device_id = 0;
    args->peak_norm = 0;
    args->sample_rate = 16000;
    args->use_proxy_metrics = 1;
    args->dump_est_pair = 0;
    snprintf(args->out_csv, sizeof(args->out_csv), "./sisdr_result.csv");
    args->dump_est_dir[0] = '\0';
    snprintf(args->tmp_dir, sizeof(args->tmp_dir), "/tmp");

    /* first two positional args */
    if (argc < 3) return -1;
    if (argv[1][0] == '-' || argv[2][0] == '-') return -1;
    snprintf(args->model,    sizeof(args->model),    "%s", argv[1]);
    snprintf(args->test_dir, sizeof(args->test_dir), "%s", argv[2]);

    for (i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--pcm_dtype") == 0 && i + 1 < argc) {
            const char *s = argv[++i];
            if      (strcmp(s, "s16") == 0 || strcmp(s, "int16") == 0) args->pcm_type = PCM_S16;
            else if (strcmp(s, "f32") == 0 || strcmp(s, "float32") == 0) args->pcm_type = PCM_F32;
            else if (strcmp(s, "s32") == 0 || strcmp(s, "int32") == 0) args->pcm_type = PCM_S32;
            else { printf("unknown --pcm_dtype: %s\n", s); return -1; }
        } else if (strcmp(argv[i], "--max_files") == 0 && i + 1 < argc) {
            args->max_files = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out_csv") == 0 && i + 1 < argc) {
            snprintf(args->out_csv, sizeof(args->out_csv), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--dump_est_dir") == 0 && i + 1 < argc) {
            snprintf(args->dump_est_dir, sizeof(args->dump_est_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--dump_est_pair") == 0 && i + 1 < argc) {
            args->dump_est_pair = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--device_id") == 0 && i + 1 < argc) {
            args->device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--input_len") == 0 && i + 1 < argc) {
            args->input_len_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--peak_norm") == 0 && i + 1 < argc) {
            args->peak_norm = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--out_layout") == 0 && i + 1 < argc) {
            const char *s = argv[++i];
            if (strcmp(s, "planar") == 0 || strcmp(s, "p") == 0) {
                args->out_layout = OUT_LAYOUT_PLANAR;
            } else if (strcmp(s, "interleaved") == 0 || strcmp(s, "i") == 0) {
                args->out_layout = OUT_LAYOUT_INTERLEAVED;
            } else if (strcmp(s, "auto") == 0 || strcmp(s, "a") == 0) {
                args->out_layout = OUT_LAYOUT_AUTO;
            } else {
                printf("unknown --out_layout: %s\n", s);
                return -1;
            }
        } else if (strcmp(argv[i], "--sample_rate") == 0 && i + 1 < argc) {
            args->sample_rate = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tmp_dir") == 0 && i + 1 < argc) {
            snprintf(args->tmp_dir, sizeof(args->tmp_dir), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--pesq_cmd") == 0 && i + 1 < argc) {
            snprintf(args->pesq_cmd, sizeof(args->pesq_cmd), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--stoi_cmd") == 0 && i + 1 < argc) {
            snprintf(args->stoi_cmd, sizeof(args->stoi_cmd), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--proxy_metrics") == 0 && i + 1 < argc) {
            args->use_proxy_metrics = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 1;
        } else {
            printf("unknown argument: %s\n", argv[i]); return -1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  main                                                                 */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    args_t    args;
    npu_ctx_t npu;
    str_vec_t mix_names = {0};
    row_vec_t rows      = {0};
    char      mix_dir[PATH_MAX], s1_dir[PATH_MAX], s2_dir[PATH_MAX];
    bool      has_s2;
    int       run_n, i, skip = 0, rc = 1;
    int       invalid_scores = 0;
    int       finite_n[5] = {0};
    FILE     *csv = NULL;
    double    mean[5] = {NAN, NAN, NAN, NAN, NAN};
    double    median[5] = {NAN, NAN, NAN, NAN, NAN};
    double    min_v[5] = {NAN, NAN, NAN, NAN, NAN};
    double    max_v[5] = {NAN, NAN, NAN, NAN, NAN};
    double    total_infer_ms = 0.0;

    memset(&npu, 0, sizeof(npu));
    npu.device_id = -1;

    rc = parse_args(argc, argv, &args);
    if (rc == 1) return 0;
    if (rc != 0) { print_usage(argv[0]); return 1; }
    if (args.sample_rate <= 0) {
        printf("[WARN] invalid --sample_rate=%d, fallback to 16000\n", args.sample_rate);
        args.sample_rate = 16000;
    }

    if (!file_exists(args.model)) {
        printf("[ERROR] model not found: %s\n", args.model); return 1;
    }
    if (!dir_exists(args.test_dir)) {
        printf("[ERROR] test_dir not found: %s\n", args.test_dir); return 1;
    }

    join_path(mix_dir, sizeof(mix_dir), args.test_dir, "mix");
    join_path(s1_dir,  sizeof(s1_dir),  args.test_dir, "s1");
    join_path(s2_dir,  sizeof(s2_dir),  args.test_dir, "s2");

    if (!dir_exists(mix_dir) || !dir_exists(s1_dir)) {
        printf("[ERROR] need subdirs: %s  %s\n", mix_dir, s1_dir); return 1;
    }
    has_s2 = dir_exists(s2_dir);

    if (list_pcm_files(mix_dir, &mix_names) || mix_names.size == 0) {
        printf("[ERROR] no .pcm files in %s\n", mix_dir);
        str_vec_free(&mix_names); return 1;
    }

    printf("model     : %s\n", args.model);
    printf("test_dir  : %s\n", args.test_dir);
    printf("pcm_dtype : %s\n",
           args.pcm_type == PCM_S16 ? "s16" :
           args.pcm_type == PCM_F32 ? "f32" : "s32");
    printf("peak_norm : %d\n", args.peak_norm);
    printf("out_layout: %s\n", layout_to_str(args.out_layout));
    printf("sample_rt : %d\n", args.sample_rate);
    printf("has_s2    : %s\n", has_s2 ? "yes" : "no (single-ref mode)");
    printf("pesq_cmd  : %s\n", args.pesq_cmd[0] ? args.pesq_cmd : "(disabled)");
    printf("stoi_cmd  : %s\n", args.stoi_cmd[0] ? args.stoi_cmd : "(disabled)");
    printf("proxy_met : %d\n", args.use_proxy_metrics);
    printf("tmp_dir   : %s\n", args.tmp_dir);
    printf("dump_est  : %s\n", args.dump_est_dir[0] ? args.dump_est_dir : "(off)");
    printf("dump_pair : %d\n", args.dump_est_pair);
    printf("mix_files : %d\n", mix_names.size);
    if ((args.pesq_cmd[0] || args.stoi_cmd[0]) && !dir_exists(args.tmp_dir)) {
        printf("[WARN] tmp_dir not found: %s, fallback to current dir\n", args.tmp_dir);
        snprintf(args.tmp_dir, sizeof(args.tmp_dir), ".");
    }
    if (args.dump_est_dir[0] && !dir_exists(args.dump_est_dir)) {
        if (mkdir(args.dump_est_dir, 0755) != 0 && !dir_exists(args.dump_est_dir)) {
            printf("[ERROR] cannot create dump_est_dir: %s\n", args.dump_est_dir);
            goto out;
        }
    }
    if (!args.pesq_cmd[0] && args.use_proxy_metrics) {
        printf("[WARN] PESQ cmd not set: using built-in PESQ proxy (not ITU-T P.862).\n");
    }
    if (!args.stoi_cmd[0] && args.use_proxy_metrics) {
        printf("[WARN] STOI cmd not set: using built-in STOI proxy (not official STOI).\n");
    }

    if (npu_init(&npu, args.model, args.device_id)) {
        printf("[ERROR] NPU init failed\n"); goto out;
    }
    if (args.input_len_override > 0) {
        npu.input_len = args.input_len_override;
        if (recalc_output_layout(&npu)) {
            printf("[ERROR] invalid --input_len\n"); goto out;
        }
    }
    printf("input_len : %d samples\n", npu.input_len);
    printf("output_ch : %d\n\n", npu.output_channels);

    run_n = mix_names.size;
    if (args.max_files > 0 && args.max_files < run_n) run_n = args.max_files;

    printf("%-6s %-20s %10s %10s %10s %7s %7s %5s %9s\n",
           "No.", "File", "SI-SDR", "SI-SDRi", "SDR", "PESQ", "STOI", "Perm", "InferMs");
    printf("%-6s %-20s %10s %10s %10s %7s %7s %5s %9s\n",
           "------", "--------------------", "----------", "----------", "----------",
           "-------", "-------", "-----", "---------");

    for (i = 0; i < run_n; ++i) {
        char   mix_path[PATH_MAX], s1_path[PATH_MAX], s2_path[PATH_MAX];
        float *mix = NULL, *s1 = NULL, *s2 = NULL;
        float *est = NULL, *est_planar = NULL, *est_inter = NULL;
        int    mix_len = 0, s1_len = 0, s2_len = 0;
        int    valid_len, channels = 0, best_perm = 0;
        int    ch_planar = 0, ch_inter = 0;
        double score = -DBL_MAX, infer_ms = 0.0;
        double sisdri = NAN, sdr = NAN, pesq = NAN, stoi = NAN;
        double infer_ms_planar = 0.0, infer_ms_inter = 0.0;
        bool   use_s2 = false;
        out_layout_t used_layout = args.out_layout;

        join_path(mix_path, sizeof(mix_path), mix_dir, mix_names.items[i]);
        join_path(s1_path,  sizeof(s1_path),  s1_dir,  mix_names.items[i]);
        join_path(s2_path,  sizeof(s2_path),  s2_dir,  mix_names.items[i]);

        if (!file_exists(s1_path)) { ++skip; continue; }

        if (read_pcm_file(mix_path, args.pcm_type, &mix, &mix_len) ||
            read_pcm_file(s1_path,  args.pcm_type, &s1,  &s1_len)) {
            free(mix); free(s1); ++skip; continue;
        }

        valid_len = mix_len < s1_len ? mix_len : s1_len;
        if (valid_len <= 0) { free(mix); free(s1); ++skip; continue; }

        if (has_s2 && file_exists(s2_path) &&
            read_pcm_file(s2_path, args.pcm_type, &s2, &s2_len) == 0) {
            use_s2 = true;
        }

        if (args.out_layout == OUT_LAYOUT_AUTO) {
            int best_perm_planar = 0, best_perm_inter = 0;
            double score_planar, score_inter;

            if (infer_wave(&npu, mix, valid_len, args.peak_norm, OUT_LAYOUT_PLANAR,
                           &est_planar, &ch_planar, &infer_ms_planar) ||
                infer_wave(&npu, mix, valid_len, args.peak_norm, OUT_LAYOUT_INTERLEAVED,
                           &est_inter, &ch_inter, &infer_ms_inter)) {
                free(mix); free(s1); free(s2); free(est_planar); free(est_inter);
                ++skip;
                continue;
            }

            score_planar = score_estimate(est_planar, ch_planar, valid_len,
                                          s1, s2, s2_len, use_s2,
                                          &best_perm_planar);
            score_inter  = score_estimate(est_inter, ch_inter, valid_len,
                                          s1, s2, s2_len, use_s2,
                                          &best_perm_inter);

            if (score_inter > score_planar) {
                est = est_inter; est_inter = NULL;
                channels = ch_inter;
                score = score_inter;
                best_perm = best_perm_inter;
                infer_ms = infer_ms_inter;
                used_layout = OUT_LAYOUT_INTERLEAVED;
            } else {
                est = est_planar; est_planar = NULL;
                channels = ch_planar;
                score = score_planar;
                best_perm = best_perm_planar;
                infer_ms = infer_ms_planar;
                used_layout = OUT_LAYOUT_PLANAR;
            }

            total_infer_ms += infer_ms;
            free(est_planar);
            free(est_inter);
        } else {
            if (infer_wave(&npu, mix, valid_len, args.peak_norm, args.out_layout,
                           &est, &channels, &infer_ms)) {
                free(mix); free(s1); free(s2);
                ++skip;
                continue;
            }
            total_infer_ms += infer_ms;
            score = score_estimate(est, channels, valid_len, s1, s2, s2_len,
                                   use_s2, &best_perm);
        }

        if (!isfinite(score)) ++invalid_scores;

        compute_extra_metrics(&args, i, mix, est, channels, valid_len, best_perm,
                              s1, s2, s2_len, use_s2, score,
                              &sisdri, &sdr, &pesq, &stoi);

        printf("%-6d %-20s %+10.4f %+10.4f %+10.4f %7.4f %7.4f %5d %9.2f\n",
               i + 1, mix_names.items[i], score, sisdri, sdr, pesq, stoi,
               best_perm, infer_ms);
        if (args.out_layout == OUT_LAYOUT_AUTO)
            printf("       selected layout: %s\n", layout_to_str(used_layout));

        if (row_vec_push(&rows, mix_names.items[i], score, sisdri, sdr, pesq, stoi,
                         best_perm, infer_ms)) {
            free(mix); free(s1); free(s2); free(est); goto out;
        }

        if (args.dump_est_dir[0] && est && valid_len > 0) {
            const float *est_s1 = est;
            const float *est_s2 = NULL;
            char stem[PATH_MAX], est_path[PATH_MAX], est_path_s2[PATH_MAX];
            if (channels >= 2) {
                const float *ch0 = est;
                const float *ch1 = est + valid_len;
                est_s1 = (best_perm == 1) ? ch1 : ch0;
                est_s2 = (best_perm == 1) ? ch0 : ch1;
            }
            snprintf(stem, sizeof(stem), "%s", mix_names.items[i]);
            {
                char *dot = strrchr(stem, '.');
                if (dot) *dot = '\0';
            }
            if (args.dump_est_pair && est_s2) {
                if (snprintf(est_path, sizeof(est_path), "%s/estimate_%s_s1.pcm",
                             args.dump_est_dir, stem) < (int)sizeof(est_path)) {
                    if (write_pcm_mono_s16(est_path, est_s1, valid_len) != 0) {
                        printf("[WARN] failed to dump estimate: %s\n", est_path);
                    }
                }
                if (snprintf(est_path_s2, sizeof(est_path_s2), "%s/estimate_%s_s2.pcm",
                             args.dump_est_dir, stem) < (int)sizeof(est_path_s2)) {
                    if (write_pcm_mono_s16(est_path_s2, est_s2, valid_len) != 0) {
                        printf("[WARN] failed to dump estimate: %s\n", est_path_s2);
                    }
                }
            } else {
                if (snprintf(est_path, sizeof(est_path), "%s/estimate_%s.pcm",
                             args.dump_est_dir, stem) < (int)sizeof(est_path)) {
                    if (write_pcm_mono_s16(est_path, est_s1, valid_len) != 0) {
                        printf("[WARN] failed to dump estimate: %s\n", est_path);
                    }
                }
            }
        }

        free(mix); free(s1); free(s2); free(est);
        mix = s1 = s2 = est = NULL;
    }

    if (rows.size == 0) { printf("\n[ERROR] no valid samples processed\n"); goto out; }

    if (calc_metric_stats(&rows, MET_SISDR, &finite_n[MET_SISDR],
                          &mean[MET_SISDR], &median[MET_SISDR],
                          &min_v[MET_SISDR], &max_v[MET_SISDR]) != 0) {
        printf("\n[ERROR] all scores are non-finite (NaN/Inf)\n");
        goto out;
    }
    (void)calc_metric_stats(&rows, MET_SISDRI, &finite_n[MET_SISDRI],
                            &mean[MET_SISDRI], &median[MET_SISDRI],
                            &min_v[MET_SISDRI], &max_v[MET_SISDRI]);
    (void)calc_metric_stats(&rows, MET_SDR, &finite_n[MET_SDR],
                            &mean[MET_SDR], &median[MET_SDR],
                            &min_v[MET_SDR], &max_v[MET_SDR]);
    (void)calc_metric_stats(&rows, MET_PESQ, &finite_n[MET_PESQ],
                            &mean[MET_PESQ], &median[MET_PESQ],
                            &min_v[MET_PESQ], &max_v[MET_PESQ]);
    (void)calc_metric_stats(&rows, MET_STOI, &finite_n[MET_STOI],
                            &mean[MET_STOI], &median[MET_STOI],
                            &min_v[MET_STOI], &max_v[MET_STOI]);

    /* ---- CSV output ---- */
    csv = fopen(args.out_csv, "w");
    if (csv) {
        fprintf(csv, "file,si_sdr_db,si_sdri_db,sdr_db,pesq,stoi,perm,infer_ms\n");
        for (i = 0; i < rows.size; ++i)
            fprintf(csv, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.3f\n",
                    rows.items[i].name, rows.items[i].sisdr,
                    rows.items[i].sisdri, rows.items[i].sdr,
                    rows.items[i].pesq, rows.items[i].stoi,
                    rows.items[i].perm, rows.items[i].infer_ms);
        fprintf(csv, "\nsummary,metric,mean,median,min,max,finite\n");
        fprintf(csv, "summary,si_sdr,%.6f,%.6f,%.6f,%.6f,%d\n",
                mean[MET_SISDR], median[MET_SISDR],
                min_v[MET_SISDR], max_v[MET_SISDR], finite_n[MET_SISDR]);
        fprintf(csv, "summary,si_sdri,%.6f,%.6f,%.6f,%.6f,%d\n",
                mean[MET_SISDRI], median[MET_SISDRI],
                min_v[MET_SISDRI], max_v[MET_SISDRI], finite_n[MET_SISDRI]);
        fprintf(csv, "summary,sdr,%.6f,%.6f,%.6f,%.6f,%d\n",
                mean[MET_SDR], median[MET_SDR],
                min_v[MET_SDR], max_v[MET_SDR], finite_n[MET_SDR]);
        fprintf(csv, "summary,pesq,%.6f,%.6f,%.6f,%.6f,%d\n",
                mean[MET_PESQ], median[MET_PESQ],
                min_v[MET_PESQ], max_v[MET_PESQ], finite_n[MET_PESQ]);
        fprintf(csv, "summary,stoi,%.6f,%.6f,%.6f,%.6f,%d\n",
                mean[MET_STOI], median[MET_STOI],
                min_v[MET_STOI], max_v[MET_STOI], finite_n[MET_STOI]);
        fclose(csv); csv = NULL;
    } else {
        printf("[WARN] could not write CSV: %s\n", args.out_csv);
    }

    /* ---- Summary ---- */
    printf("\n========== Separation Metrics Summary ==========\n");
    printf("evaluated     : %d\n", rows.size);
    printf("skipped       : %d\n", skip);
    printf("invalid_si_sdr: %d\n", invalid_scores);
    printf("SI-SDR  mean/median/min/max: %+.4f / %+.4f / %+.4f / %+.4f dB (n=%d)\n",
           mean[MET_SISDR], median[MET_SISDR], min_v[MET_SISDR], max_v[MET_SISDR],
           finite_n[MET_SISDR]);
    printf("SI-SDRi mean/median/min/max: %+.4f / %+.4f / %+.4f / %+.4f dB (n=%d)\n",
           mean[MET_SISDRI], median[MET_SISDRI], min_v[MET_SISDRI], max_v[MET_SISDRI],
           finite_n[MET_SISDRI]);
    printf("SDR     mean/median/min/max: %+.4f / %+.4f / %+.4f / %+.4f dB (n=%d)\n",
           mean[MET_SDR], median[MET_SDR], min_v[MET_SDR], max_v[MET_SDR],
           finite_n[MET_SDR]);
    printf("PESQ    mean/median/min/max: %+.4f / %+.4f / %+.4f / %+.4f (n=%d)\n",
           mean[MET_PESQ], median[MET_PESQ], min_v[MET_PESQ], max_v[MET_PESQ],
           finite_n[MET_PESQ]);
    printf("STOI    mean/median/min/max: %+.4f / %+.4f / %+.4f / %+.4f (n=%d)\n",
           mean[MET_STOI], median[MET_STOI], min_v[MET_STOI], max_v[MET_STOI],
           finite_n[MET_STOI]);
    if (args.pesq_cmd[0] && finite_n[MET_PESQ] == 0)
        printf("[WARN] PESQ command returned no valid values. Check --pesq_cmd output format.\n");
    if (args.stoi_cmd[0] && finite_n[MET_STOI] == 0)
        printf("[WARN] STOI command returned no valid values. Check --stoi_cmd output format.\n");
    printf("infer_avg : %.2f ms/file\n",
           rows.size > 0 ? total_infer_ms / rows.size : 0.0);
    printf("infer_tot : %.2f s\n",   total_infer_ms / 1000.0);
    printf("CSV saved : %s\n",       args.out_csv);
    printf("=====================================\n");

    rc = 0;
out:
    if (csv) fclose(csv);
    row_vec_free(&rows);
    str_vec_free(&mix_names);
    if (npu.initialized || npu.model_id || npu.context)
        npu_deinit(&npu);
    return rc;
}
