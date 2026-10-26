#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "rknn_api.h"

#define SAMPLES 32000
#define PCM_BYTES (SAMPLES * sizeof(int16_t))
#define PATH_SIZE 1024

static double elapsed_ms(const struct timespec *begin, const struct timespec *end) {
    return (double)(end->tv_sec - begin->tv_sec) * 1000.0 +
           (double)(end->tv_nsec - begin->tv_nsec) / 1000000.0;
}

static int has_pcm_suffix(const char *name) {
    size_t len = strlen(name);
    return len > 4 && strcmp(name + len - 4, ".pcm") == 0;
}

static int make_dir(const char *path) {
    if (mkdir(path, 0755) == 0 || errno == EEXIST) return 0;
    fprintf(stderr, "mkdir %s: %s\n", path, strerror(errno));
    return -1;
}

static int read_pcm16(const char *path, int16_t *samples) {
    FILE *fp = fopen(path, "rb");
    long size;
    if (!fp) {
        fprintf(stderr, "fopen %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0 || (size = ftell(fp)) != PCM_BYTES ||
        fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s must be exactly %zu bytes\n", path, (size_t)PCM_BYTES);
        fclose(fp);
        return -1;
    }
    if (fread(samples, sizeof(int16_t), SAMPLES, fp) != SAMPLES) {
        fprintf(stderr, "read %s failed\n", path);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int write_pcm16(const char *path, const float *samples) {
    FILE *fp = fopen(path, "wb");
    int i;
    if (!fp) {
        fprintf(stderr, "fopen %s: %s\n", path, strerror(errno));
        return -1;
    }
    for (i = 0; i < SAMPLES; ++i) {
        float x = samples[i];
        int16_t value;
        /* This model outputs waveform samples directly in signed PCM16 scale. */
        if (x > 32767.0f) x = 32767.0f;
        if (x < -32768.0f) x = -32768.0f;
        value = (int16_t)lrintf(x);
        if (fwrite(&value, sizeof(value), 1, fp) != 1) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

/* Scale-invariant SDR in dB. Both signals are centered before projection. */
static double si_sdr(const float *estimate, const int16_t *reference) {
    double mean_est = 0.0, mean_ref = 0.0;
    double dot = 0.0, ref_energy = 0.0, est_energy = 0.0;
    double target_energy, noise_energy, scale;
    int i;
    for (i = 0; i < SAMPLES; ++i) {
        mean_est += estimate[i];
        mean_ref += (double)reference[i] / 32768.0;
    }
    mean_est /= SAMPLES;
    mean_ref /= SAMPLES;
    for (i = 0; i < SAMPLES; ++i) {
        double e = estimate[i] - mean_est;
        double r = (double)reference[i] / 32768.0 - mean_ref;
        dot += e * r;
        ref_energy += r * r;
        est_energy += e * e;
    }
    if (ref_energy < 1e-20 || est_energy < 1e-20) return -100.0;
    scale = dot / ref_energy;
    target_energy = scale * scale * ref_energy;
    noise_energy = est_energy - target_energy;
    if (noise_energy < 1e-20) noise_energy = 1e-20;
    if (target_energy < 1e-20) return -100.0;
    return 10.0 * log10(target_energy / noise_energy);
}

static double si_sdr_pcm(const int16_t *estimate, const int16_t *reference) {
    float *tmp = malloc(SAMPLES * sizeof(*tmp));
    double score;
    int i;
    if (!tmp) return -100.0;
    for (i = 0; i < SAMPLES; ++i) tmp[i] = (float)estimate[i] / 32768.0f;
    score = si_sdr(tmp, reference);
    free(tmp);
    return score;
}

int main(int argc, char **argv) {
    const char *model_path, *mix_dir, *s1_dir, *s2_dir, *out_dir;
    char out_s1_dir[PATH_SIZE], out_s2_dir[PATH_SIZE], log_path[PATH_SIZE];
    unsigned char *model = NULL;
    long model_size;
    FILE *model_fp = NULL, *log_fp = NULL;
    rknn_context ctx = 0;
    rknn_input_output_num io_num;
    rknn_input input_desc;
    rknn_output outputs[1];
    DIR *dir = NULL;
    struct dirent *entry;
    int16_t mix[SAMPLES], ref1[SAMPLES], ref2[SAMPLES];
    float input[SAMPLES];
    unsigned int count = 0;
    double total_run_ms = 0.0, total_sisdr = 0.0, total_sisdri = 0.0;
    int ret;

    if (argc != 6) {
        fprintf(stderr, "Usage: %s model.rknn mix_dir s1_dir s2_dir output_dir\n", argv[0]);
        return 2;
    }
    model_path = argv[1];
    mix_dir = argv[2];
    s1_dir = argv[3];
    s2_dir = argv[4];
    out_dir = argv[5];

    model_fp = fopen(model_path, "rb");
    if (!model_fp || fseek(model_fp, 0, SEEK_END) != 0 || (model_size = ftell(model_fp)) <= 0 ||
        fseek(model_fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Cannot read model %s\n", model_path);
        if (model_fp) fclose(model_fp);
        return 1;
    }
    model = malloc((size_t)model_size);
    if (!model || fread(model, 1, (size_t)model_size, model_fp) != (size_t)model_size) {
        fprintf(stderr, "Model read failed\n");
        fclose(model_fp);
        free(model);
        return 1;
    }
    fclose(model_fp);

    ret = rknn_init(&ctx, model, (uint32_t)model_size, 0, NULL);
    free(model);
    if (ret < 0) {
        fprintf(stderr, "rknn_init failed: %d\n", ret);
        return 1;
    }
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input != 1 || io_num.n_output != 1) {
        fprintf(stderr, "Unexpected model IO: ret=%d inputs=%u outputs=%u\n", ret, io_num.n_input, io_num.n_output);
        rknn_destroy(ctx);
        return 1;
    }
    printf("Model initialized: inputs=%u outputs=%u\n", io_num.n_input, io_num.n_output);

    if (make_dir(out_dir) != 0) goto fail;
    snprintf(out_s1_dir, sizeof(out_s1_dir), "%s/s1", out_dir);
    snprintf(out_s2_dir, sizeof(out_s2_dir), "%s/s2", out_dir);
    snprintf(log_path, sizeof(log_path), "%s/metrics.csv", out_dir);
    if (make_dir(out_s1_dir) != 0 || make_dir(out_s2_dir) != 0) goto fail;
    log_fp = fopen(log_path, "w");
    if (!log_fp) goto fail;
    fprintf(log_fp, "file,npu_ms,rtf,si_sdr_db,si_sdri_db,assignment\n");

    dir = opendir(mix_dir);
    if (!dir) {
        fprintf(stderr, "opendir %s: %s\n", mix_dir, strerror(errno));
        goto fail;
    }
    while ((entry = readdir(dir)) != NULL) {
        char mix_path[PATH_SIZE], ref1_path[PATH_SIZE], ref2_path[PATH_SIZE];
        char out1_path[PATH_SIZE], out2_path[PATH_SIZE];
        struct timespec run_begin, run_end;
        const float *raw1, *raw2;
        double score_a1, score_a2, score_b1, score_b2;
        double base1, base2, avg_a, avg_b, avg, sdri;
        int swap;
        int i;

        if (!has_pcm_suffix(entry->d_name)) continue;
        snprintf(mix_path, sizeof(mix_path), "%s/%s", mix_dir, entry->d_name);
        snprintf(ref1_path, sizeof(ref1_path), "%s/%s", s1_dir, entry->d_name);
        snprintf(ref2_path, sizeof(ref2_path), "%s/%s", s2_dir, entry->d_name);
        if (read_pcm16(mix_path, mix) || read_pcm16(ref1_path, ref1) || read_pcm16(ref2_path, ref2)) {
            fprintf(stderr, "Skipping %s\n", entry->d_name);
            continue;
        }
        for (i = 0; i < SAMPLES; ++i) input[i] = (float)mix[i] / 32768.0f;
        memset(&input_desc, 0, sizeof(input_desc));
        input_desc.index = 0;
        input_desc.type = RKNN_TENSOR_FLOAT32;
        input_desc.fmt = RKNN_TENSOR_NCHW;
        input_desc.size = sizeof(input);
        input_desc.buf = input;
        if ((ret = rknn_inputs_set(ctx, 1, &input_desc)) < 0) {
            fprintf(stderr, "rknn_inputs_set %s failed: %d\n", entry->d_name, ret);
            goto fail;
        }
        clock_gettime(CLOCK_MONOTONIC, &run_begin);
        if ((ret = rknn_run(ctx, NULL)) < 0) {
            fprintf(stderr, "rknn_run %s failed: %d\n", entry->d_name, ret);
            goto fail;
        }
        clock_gettime(CLOCK_MONOTONIC, &run_end);
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        if ((ret = rknn_outputs_get(ctx, 1, outputs, NULL)) < 0) {
            fprintf(stderr, "rknn_outputs_get %s failed: %d\n", entry->d_name, ret);
            goto fail;
        }
        if (outputs[0].size < 2 * SAMPLES * sizeof(float)) {
            fprintf(stderr, "Output for %s is too small: %u bytes\n", entry->d_name, outputs[0].size);
            rknn_outputs_release(ctx, 1, outputs);
            goto fail;
        }
        raw1 = (const float *)outputs[0].buf;
        raw2 = raw1 + SAMPLES;
        score_a1 = si_sdr(raw1, ref1);
        score_a2 = si_sdr(raw2, ref2);
        score_b1 = si_sdr(raw1, ref2);
        score_b2 = si_sdr(raw2, ref1);
        avg_a = (score_a1 + score_a2) / 2.0;
        avg_b = (score_b1 + score_b2) / 2.0;
        swap = avg_b > avg_a;
        avg = swap ? avg_b : avg_a;
        base1 = si_sdr_pcm(mix, ref1);
        base2 = si_sdr_pcm(mix, ref2);
        sdri = avg - (base1 + base2) / 2.0;
        snprintf(out1_path, sizeof(out1_path), "%s/%s", out_s1_dir, entry->d_name);
        snprintf(out2_path, sizeof(out2_path), "%s/%s", out_s2_dir, entry->d_name);
        if (write_pcm16(out1_path, swap ? raw2 : raw1) || write_pcm16(out2_path, swap ? raw1 : raw2)) {
            rknn_outputs_release(ctx, 1, outputs);
            goto fail;
        }
        rknn_outputs_release(ctx, 1, outputs);

        total_run_ms += elapsed_ms(&run_begin, &run_end);
        total_sisdr += avg;
        total_sisdri += sdri;
        ++count;
        fprintf(log_fp, "%s,%.3f,%.6f,%.4f,%.4f,%s\n", entry->d_name,
                elapsed_ms(&run_begin, &run_end), elapsed_ms(&run_begin, &run_end) / 2000.0,
                avg, sdri, swap ? "raw2->s1" : "raw1->s1");
        printf("%4u %-24s npu=%7.3f ms RTF=%.4f SI-SDR=%.2f dB SI-SDRi=%.2f dB\n",
               count, entry->d_name, elapsed_ms(&run_begin, &run_end),
               elapsed_ms(&run_begin, &run_end) / 2000.0, avg, sdri);
    }
    closedir(dir);
    fclose(log_fp);
    if (count == 0) {
        fprintf(stderr, "No PCM files evaluated\n");
        rknn_destroy(ctx);
        return 1;
    }
    printf("\n========== Summary ==========\n");
    printf("Files:              %u\n", count);
    printf("Average NPU time:   %.3f ms / 2-second segment\n", total_run_ms / count);
    printf("Average RTF:        %.6f\n", total_run_ms / count / 2000.0);
    printf("Mean PIT SI-SDR:    %.4f dB\n", total_sisdr / count);
    printf("Mean PIT SI-SDRi:   %.4f dB\n", total_sisdri / count);
    printf("Metrics CSV:        %s\n", log_path);
    rknn_destroy(ctx);
    return 0;

fail:
    if (dir) closedir(dir);
    if (log_fp) fclose(log_fp);
    rknn_destroy(ctx);
    return 1;
}
