#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

struct futhark_context_config {
    int device;
    int platform;
    size_t default_group_size;
    size_t default_num_groups;
    size_t default_tile_size;
    int profiling;
};

struct futhark_context {
    struct futhark_context_config *cfg;
    void *opencl_ctx;
    void *error;
};

struct futhark_f32_1d {
    float *data;
    int64_t shape[1];
};

struct futhark_f32_2d {
    float *data;
    int64_t shape[2];
};

struct futhark_f32_3d {
    float *data;
    int64_t shape[3];
};

struct futhark_u64_1d {
    uint64_t *data;
    int64_t shape[1];
};

struct futhark_i64_1d {
    int64_t *data;
    int64_t shape[1];
};

struct futhark_context_config *futhark_context_config_new(void) {
    struct futhark_context_config *cfg = malloc(sizeof(struct futhark_context_config));
    if (cfg) {
        cfg->device = 0;
        cfg->platform = 0;
        cfg->default_group_size = 256;
        cfg->default_num_groups = 128;
        cfg->default_tile_size = 16;
        cfg->profiling = 0;
    }
    return cfg;
}

void futhark_context_config_free(struct futhark_context_config *cfg) {
    free(cfg);
}

void futhark_context_config_set_device(struct futhark_context_config *cfg, int device) {
    if (cfg) cfg->device = device;
}

void futhark_context_config_set_platform(struct futhark_context_config *cfg, int platform) {
    if (cfg) cfg->platform = platform;
}

struct futhark_context *futhark_context_new(struct futhark_context_config *cfg) {
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    if (ctx) {
        ctx->cfg = cfg;
        ctx->opencl_ctx = NULL;
        ctx->error = NULL;
    }
    return ctx;
}

void futhark_context_free(struct futhark_context *ctx) {
    if (ctx) {
        free(ctx);
    }
}

int futhark_context_sync(struct futhark_context *ctx) {
    (void)ctx;
    return 0;
}

char *futhark_context_get_error(struct futhark_context *ctx) {
    return ctx ? ctx->error : NULL;
}

struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const float *data, int64_t dim0) {
    (void)ctx;
    struct futhark_f32_1d *arr = malloc(sizeof(struct futhark_f32_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(float));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(float));
        }
    }
    return arr;
}

struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->data = malloc(dim0 * dim1 * sizeof(float));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * dim1 * sizeof(float));
        }
    }
    return arr;
}

struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    struct futhark_f32_3d *arr = malloc(sizeof(struct futhark_f32_3d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->shape[2] = dim2;
        arr->data = malloc(dim0 * dim1 * dim2 * sizeof(float));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * dim1 * dim2 * sizeof(float));
        }
    }
    return arr;
}

struct futhark_u64_1d *futhark_new_u64_1d(struct futhark_context *ctx, const uint64_t *data, int64_t dim0) {
    (void)ctx;
    struct futhark_u64_1d *arr = malloc(sizeof(struct futhark_u64_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(uint64_t));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(uint64_t));
        }
    }
    return arr;
}

struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0) {
    (void)ctx;
    struct futhark_i64_1d *arr = malloc(sizeof(struct futhark_i64_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(int64_t));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(int64_t));
        }
    }
    return arr;
}

void futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int futhark_values_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(float));
        return 0;
    }
    return 1;
}

int futhark_values_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * arr->shape[1] * sizeof(float));
        return 0;
    }
    return 1;
}

int futhark_values_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * arr->shape[1] * arr->shape[2] * sizeof(float));
        return 0;
    }
    return 1;
}

int futhark_values_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr, uint64_t *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(uint64_t));
        return 0;
    }
    return 1;
}

int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(int64_t));
        return 0;
    }
    return 1;
}

int futhark_entry_matmul(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_f32_2d *a, const struct futhark_f32_2d *b) {
    (void)ctx;
    if (!a || !b || !out) return 1;
    
    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];
    
    if (k != b->shape[0]) return 1;
    
    *out = malloc(sizeof(struct futhark_f32_2d));
    if (!*out) return 1;
    
    (*out)->shape[0] = m;
    (*out)->shape[1] = n;
    (*out)->data = calloc(m * n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t kk = 0; kk < k; kk++) {
                sum += a->data[i * k + kk] * b->data[kk * n + j];
            }
            (*out)->data[i * n + j] = sum;
        }
    }
    
    return 0;
}

int futhark_entry_batch_matmul(struct futhark_context *ctx, struct futhark_f32_3d **out, const struct futhark_f32_3d *a, const struct futhark_f32_3d *c) {
    (void)ctx;
    if (!a || !c || !out) return 1;
    
    int64_t batch = a->shape[0];
    int64_t m = a->shape[1];
    int64_t k = a->shape[2];
    int64_t n = c->shape[2];
    
    if (batch != c->shape[0] || k != c->shape[1]) return 1;
    
    *out = malloc(sizeof(struct futhark_f32_3d));
    if (!*out) return 1;
    
    (*out)->shape[0] = batch;
    (*out)->shape[1] = m;
    (*out)->shape[2] = n;
    (*out)->data = calloc(batch * m * n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int64_t kk = 0; kk < k; kk++) {
                    sum += a->data[b * m * k + i * k + kk] * c->data[b * k * n + kk * n + j];
                }
                (*out)->data[b * m * n + i * n + j] = sum;
            }
        }
    }
    
    return 0;
}

int futhark_entry_dot(struct futhark_context *ctx, float *out, const struct futhark_f32_1d *a, const struct futhark_f32_1d *b) {
    (void)ctx;
    if (!a || !b || !out || a->shape[0] != b->shape[0]) return 1;
    
    float sum = 0.0f;
    for (int64_t i = 0; i < a->shape[0]; i++) {
        sum += a->data[i] * b->data[i];
    }
    *out = sum;
    
    return 0;
}

int futhark_entry_apply_softmax(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;
    
    int64_t n = x->shape[0];
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    float max_val = x->data[0];
    for (int64_t i = 1; i < n; i++) {
        if (x->data[i] > max_val) max_val = x->data[i];
    }
    
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = expf(x->data[i] - max_val);
        sum += (*out)->data[i];
    }
    
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] /= sum;
    }
    
    return 0;
}

int futhark_entry_apply_layer_norm(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x, const struct futhark_f32_1d *gamma, const struct futhark_f32_1d *beta, float eps) {
    (void)ctx;
    if (!x || !gamma || !beta || !out) return 1;
    
    int64_t n = x->shape[0];
    if (gamma->shape[0] != n || beta->shape[0] != n) return 1;
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    float mean = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        mean += x->data[i];
    }
    mean /= (float)n;
    
    float variance = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = x->data[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)n;
    
    float std_dev = sqrtf(variance + eps);
    
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = gamma->data[i] * ((x->data[i] - mean) / std_dev) + beta->data[i];
    }
    
    return 0;
}

int futhark_entry_apply_relu(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;
    
    int64_t n = x->shape[0];
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    }
    
    return 0;
}

int futhark_entry_apply_gelu(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;
    
    int64_t n = x->shape[0];
    const float sqrt_2_over_pi = 0.7978845608f;
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t i = 0; i < n; i++) {
        float xi = x->data[i];
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi)));
        (*out)->data[i] = xi * cdf;
    }
    
    return 0;
}

int futhark_entry_clip_fisher(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *fisher, float clip_val) {
    (void)ctx;
    if (!fisher || !out) return 1;
    
    int64_t n = fisher->shape[0];
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = fisher->data[i] > clip_val ? fisher->data[i] : clip_val;
    }
    
    return 0;
}

int futhark_entry_reduce_gradients(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_2d *gradients) {
    (void)ctx;
    if (!gradients || !out) return 1;
    
    int64_t batch = gradients->shape[0];
    int64_t n = gradients->shape[1];
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = calloc(n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < n; i++) {
            (*out)->data[i] += gradients->data[b * n + i];
        }
    }
    
    return 0;
}

int futhark_entry_rank_segments(struct futhark_context *ctx, struct futhark_f32_1d **out, uint64_t query_hash, const struct futhark_u64_1d *segment_hashes, const struct futhark_f32_1d *base_scores) {
    (void)ctx;
    if (!segment_hashes || !base_scores || !out) return 1;
    
    int64_t n = segment_hashes->shape[0];
    if (base_scores->shape[0] != n) return 1;
    
    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t i = 0; i < n; i++) {
        float match_bonus = (segment_hashes->data[i] == query_hash) ? 1.0f : 0.0f;
        (*out)->data[i] = base_scores->data[i] + match_bonus;
    }
    
    return 0;
}

int futhark_entry_select_topk(struct futhark_context *ctx, struct futhark_f32_1d **out_scores, struct futhark_i64_1d **out_indices, int64_t k, const struct futhark_f32_1d *scores) {
    (void)ctx;
    if (!scores || !out_scores || !out_indices) return 1;
    
    int64_t n = scores->shape[0];
    if (k > n) k = n;
    
    typedef struct {
        float score;
        int64_t index;
    } ScoreIndex;
    
    ScoreIndex *pairs = malloc(n * sizeof(ScoreIndex));
    if (!pairs) return 1;
    
    for (int64_t i = 0; i < n; i++) {
        pairs[i].score = scores->data[i];
        pairs[i].index = i;
    }
    
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = i + 1; j < n; j++) {
            if (pairs[j].score > pairs[i].score) {
                ScoreIndex temp = pairs[i];
                pairs[i] = pairs[j];
                pairs[j] = temp;
            }
        }
    }
    
    *out_scores = malloc(sizeof(struct futhark_f32_1d));
    *out_indices = malloc(sizeof(struct futhark_i64_1d));
    
    if (!*out_scores || !*out_indices) {
        free(pairs);
        return 1;
    }
    
    (*out_scores)->shape[0] = k;
    (*out_scores)->data = malloc(k * sizeof(float));
    (*out_indices)->shape[0] = k;
    (*out_indices)->data = malloc(k * sizeof(int64_t));
    
    if (!(*out_scores)->data || !(*out_indices)->data) {
        free(pairs);
        free(*out_scores);
        free(*out_indices);
        return 1;
    }
    
    for (int64_t i = 0; i < k; i++) {
        (*out_scores)->data[i] = pairs[i].score;
        (*out_indices)->data[i] = pairs[i].index;
    }
    
    free(pairs);
    return 0;
}
