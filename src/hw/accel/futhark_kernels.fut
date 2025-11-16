-- JAIDE v40 Futhark GPU Kernels
-- Optimized kernels for tensor operations with complete entry points

-- Matrix multiplication with tiling
let matmul_tiled [m][n][k] (a: [m][k]f32) (b: [k][n]f32): [m][n]f32 =
  let tile_size = 16i64
  in map (\i ->
    map (\j ->
      let tiles = k / tile_size
      in reduce (+) 0f32
        (map (\t ->
          reduce (+) 0f32
            (map (\kk ->
              a[i, t*tile_size + kk] * b[t*tile_size + kk, j]
            ) (iota tile_size))
        ) (iota tiles))
    ) (iota n)
  ) (iota m)

-- Batched matrix multiplication
let batched_matmul [b][m][n][k] (a: [b][m][k]f32) (c: [b][k][n]f32): [b][m][n]f32 =
  map2 (\a_mat c_mat -> matmul_tiled a_mat c_mat) a c

-- Vector dot product with reduction
let dot_product [n] (a: [n]f32) (b: [n]f32): f32 =
  reduce (+) 0f32 (map2 (*) a b)

-- Softmax with numerical stability
let softmax [n] (x: [n]f32): [n]f32 =
  let max_val = reduce f32.max (-f32.inf) x
  let exp_x = map (\xi -> f32.exp (xi - max_val)) x
  let sum = reduce (+) 0f32 exp_x
  in map (/ sum) exp_x

-- Layer normalization
let layer_norm [n] (x: [n]f32) (gamma: [n]f32) (beta: [n]f32) (eps: f32): [n]f32 =
  let mean = (reduce (+) 0f32 x) / f32.i64 n
  let variance = (reduce (+) 0f32 (map (\xi -> (xi - mean) * (xi - mean)) x)) / f32.i64 n
  let std_dev = f32.sqrt (variance + eps)
  in map3 (\xi g b -> g * ((xi - mean) / std_dev) + b) x gamma beta

-- ReLU activation
let relu [n] (x: [n]f32): [n]f32 =
  map (\xi -> f32.max 0f32 xi) x

-- GELU activation
let gelu [n] (x: [n]f32): [n]f32 =
  let sqrt_2_over_pi = 0.7978845608f32
  in map (\xi ->
    let cdf = 0.5f32 * (1.0f32 + f32.tanh (sqrt_2_over_pi * (xi + 0.044715f32 * xi * xi * xi)))
    in xi * cdf
  ) x

-- Spectral clipping for Fisher diagonal
let spectral_clip [n] (fisher: [n]f32) (clip_val: f32): [n]f32 =
  map (\f -> f32.max f clip_val) fisher

-- Batch reduction for gradient accumulation
let batch_reduce [b][n] (gradients: [b][n]f32): [n]f32 =
  reduce_comm (\a b -> map2 (+) a b) (replicate n 0f32) gradients

-- Segment scoring (for ranker)
let score_segments [n] (query_hash: u64) (segment_hashes: [n]u64) (base_scores: [n]f32): [n]f32 =
  map2 (\hash score ->
    let match_bonus = if hash == query_hash then 1.0f32 else 0.0f32
    in score + match_bonus
  ) segment_hashes base_scores

-- Top-K selection using radix sort
let topk [n] (k: i64) (scores: [n]f32) (indices: [n]i64): ([k]f32, [k]i64) =
  let sorted_pairs = zip scores indices
                      |> radix_sort_by_key (.0) (>)
  let top = take k sorted_pairs
  in (map (.0) top, map (.1) top)

-- RSF scatter operation
let rsf_scatter [n] (x: [n]f32) (indices: [n]i64): [n]f32 =
  let half = n / 2
  in map (\i ->
    if i < half then
      let j = indices[i] % half
      in x[j] + x[j + half]
    else
      let j = indices[i - half] % half
      in x[j] - x[j + half]
  ) (iota n)

-- RSF flow operation
let rsf_flow [n] (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32) (s_bias: [n]f32) (t_bias: [n]f32): [n]f32 =
  let half = n / 2
  let x_s = map (\i -> x[i] * s_weight[i] + s_bias[i]) (iota half)
  let x_t = map (\i -> x[i + half] * t_weight[i] + t_bias[i]) (iota half)
  let combined = map2 (+) x_s x_t
  in scatter (replicate n 0f32) (iota n) (map (\i -> if i < half then combined[i] else combined[i - half]) (iota n))

-- RSF forward layer
let rsf_forward_layer [n] (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32) (s_bias: [n]f32) (t_bias: [n]f32) (perm_indices: [n]i64): [n]f32 =
  let scattered = rsf_scatter x perm_indices
  in rsf_flow scattered s_weight t_weight s_bias t_bias

-- RSF backward scatter
let rsf_backward_scatter [n] (grad: [n]f32) (indices: [n]i64): [n]f32 =
  let half = n / 2
  in map (\i ->
    if i < half then
      let j = indices[i] % half
      in grad[j] + grad[j + half]
    else
      let j = indices[i - half] % half
      in grad[j] - grad[j + half]
  ) (iota n)

-- RSF backward flow
let rsf_backward_flow [n] (grad_out: [n]f32) (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32): ([n]f32, [n]f32, [n]f32, [n]f32, [n]f32) =
  let half = n / 2
  let grad_s_bias = map (\i -> grad_out[i]) (iota half)
  let grad_t_bias = map (\i -> grad_out[i + half]) (iota half)
  let grad_s_weight = map2 (\g xi -> g * xi) grad_s_bias (map (\i -> x[i]) (iota half))
  let grad_t_weight = map2 (\g xi -> g * xi) grad_t_bias (map (\i -> x[i + half]) (iota half))
  let grad_x_s = map2 (*) grad_s_bias s_weight
  let grad_x_t = map2 (*) grad_t_bias t_weight
  let grad_x = map (\i -> if i < half then grad_x_s[i] else grad_x_t[i - half]) (iota n)
  let grad_s_weight_full = map (\i -> if i < half then grad_s_weight[i] else 0f32) (iota n)
  let grad_t_weight_full = map (\i -> if i < half then grad_t_weight[i] else 0f32) (iota n)
  let grad_s_bias_full = map (\i -> if i < half then grad_s_bias[i] else 0f32) (iota n)
  let grad_t_bias_full = map (\i -> if i < half then grad_t_bias[i] else 0f32) (iota n)
  in (grad_x, grad_s_weight_full, grad_t_weight_full, grad_s_bias_full, grad_t_bias_full)

-- RSF backward layer
let rsf_backward_layer [n] (grad_out: [n]f32) (x: [n]f32) (s_weight: [n]f32) (t_weight: [n]f32) (perm_indices: [n]i64): ([n]f32, [n]f32, [n]f32, [n]f32, [n]f32) =
  let (grad_x_flow, grad_s_w, grad_t_w, grad_s_b, grad_t_b) = rsf_backward_flow grad_out x s_weight t_weight
  let grad_x = rsf_backward_scatter grad_x_flow perm_indices
  in (grad_x, grad_s_w, grad_t_w, grad_s_b, grad_t_b)

-- Hash sequence
let hash_sequence [m] (tokens: [m]u32): u64 =
  let multiplier = 31u64
  in reduce (\h t -> h * multiplier + u64.u32 t) 0u64 tokens

-- SSI hash insert
let ssi_hash_insert [n] (hashes: [n]u64) (new_hash: u64): [n+1]u64 =
  let pos = reduce (+) 0i64 (map (\h -> if h < new_hash then 1i64 else 0i64) hashes)
  in scatter (replicate (n+1) 0u64)
             (map (\i -> if i < pos then i else i + 1) (iota n))
             hashes ++ [new_hash]

-- SSI search
let ssi_search [n][m] (tree_hashes: [n]u64) (query: [m]u32): i64 =
  let query_hash = hash_sequence query
  let distances = map (\h ->
    let diff = if h > query_hash then h - query_hash else query_hash - h
    in diff
  ) tree_hashes
  let min_dist = reduce f64.min f64.inf (map f64.u64 distances)
  let min_idx = reduce (\acc i ->
    if f64.u64 distances[i] == min_dist then i else acc
  ) 0i64 (iota n)
  in min_idx

-- SSI retrieve top-k
let ssi_retrieve_topk [n][m] (tree_hashes: [n]u64) (scores: [n]f32) (query: [m]u32) (k: i64): ([k]u64, [k]f32) =
  let query_hash = hash_sequence query
  let adjusted_scores = map2 (\h score ->
    let match_bonus = if h == query_hash then 10.0f32 else 0.0f32
    let proximity = 1.0f32 / (1.0f32 + f32.u64 (if h > query_hash then h - query_hash else query_hash - h))
    in score + match_bonus + proximity
  ) tree_hashes scores
  let sorted_indices = radix_sort_by_key (\i -> adjusted_scores[i]) (>) (iota n)
  let top_indices = take k sorted_indices
  let top_hashes = map (\i -> tree_hashes[i]) top_indices
  let top_scores = map (\i -> adjusted_scores[i]) top_indices
  in (top_hashes, top_scores)

-- SSI compute similarity
let ssi_compute_similarity [m] (query: [m]u32) (candidate: [m]u32): f32 =
  let matches = reduce (+) 0i64 (map2 (\q c -> if q == c then 1i64 else 0i64) query candidate)
  let max_len = i64.max (i64.i32 m) (i64.i32 m)
  in f32.i64 matches / f32.i64 max_len

-- N-gram hash
let ngram_hash [n] (tokens: [n]u32) (ngram_size: i64): []u64 =
  let num_ngrams = n - ngram_size + 1
  in map (\i ->
    let ngram = tokens[i:i+ngram_size]
    in hash_sequence ngram
  ) (iota num_ngrams)

-- LSH hash
let lsh_hash [n] (vec: [n]f32) (num_tables: i64) (seed: u64): [num_tables]u64 =
  map (\table_idx ->
    let table_seed = seed + u64.i64 table_idx
    let proj = reduce (+) 0f32 (map2 (\v i ->
      let pseudo_rand = f32.u64 ((table_seed + u64.i64 i) * 2654435761u64)
      in v * pseudo_rand
    ) vec (iota n))
    in if proj > 0f32 then 1u64 else 0u64
  ) (iota num_tables)

-- Fisher diagonal update
let fisher_diagonal_update [n] (fisher: [n]f32) (gradient: [n]f32) (decay: f32): [n]f32 =
  map2 (\f g -> decay * f + (1.0f32 - decay) * g * g) fisher gradient

-- Spectral natural gradient
let spectral_natural_gradient [n] (gradient: [n]f32) (fisher: [n]f32) (damping: f32): [n]f32 =
  map2 (\g f -> g / (f + damping)) gradient fisher

-- Attention mechanism
let attention [seq_len][d_model] (query: [seq_len][d_model]f32) (key: [seq_len][d_model]f32) (value: [seq_len][d_model]f32): [seq_len][d_model]f32 =
  let scores = map (\q -> map (\k -> dot_product q k) key) query
  let scaled_scores = map (\row -> map (/ f32.sqrt (f32.i64 d_model)) row) scores
  let attention_weights = map softmax scaled_scores
  in map (\weights -> reduce_comm (map2 (\w v -> map (* w) v)) (replicate d_model 0f32) (zip weights value)) attention_weights

-- Convolution 1D
let conv1d [input_len][kernel_size] (input: [input_len]f32) (kernel: [kernel_size]f32): [input_len - kernel_size + 1]f32 =
  map (\i ->
    reduce (+) 0f32 (map2 (*) (input[i:i+kernel_size]) kernel)
  ) (iota (input_len - kernel_size + 1))

-- Max pooling 1D
let maxpool1d [input_len] (input: [input_len]f32) (pool_size: i64): [input_len / pool_size]f32 =
  map (\i ->
    let pool_start = i * pool_size
    let pool_end = pool_start + pool_size
    in reduce f32.max (-f32.inf) input[pool_start:pool_end]
  ) (iota (input_len / pool_size))

-- Element-wise operations
let elem_add [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (+) a b
let elem_mul [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (*) a b
let elem_div [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (/) a b
let elem_sub [n] (a: [n]f32) (b: [n]f32): [n]f32 = map2 (-) a b

-- Scalar operations
let scalar_add [n] (a: [n]f32) (s: f32): [n]f32 = map (+ s) a
let scalar_mul [n] (a: [n]f32) (s: f32): [n]f32 = map (* s) a
let scalar_div [n] (a: [n]f32) (s: f32): [n]f32 = map (/ s) a

-- Reduction operations
let sum [n] (x: [n]f32): f32 = reduce (+) 0f32 x
let mean [n] (x: [n]f32): f32 = (reduce (+) 0f32 x) / f32.i64 n
let max [n] (x: [n]f32): f32 = reduce f32.max (-f32.inf) x
let min [n] (x: [n]f32): f32 = reduce f32.min f32.inf x

-- ENTRY POINTS FOR C FFI

-- Basic tensor operations
entry matmul [m][n][k] (a: [m][k]f32) (b: [k][n]f32): [m][n]f32 = matmul_tiled a b
entry batch_matmul [b][m][n][k] (a: [b][m][k]f32) (c: [b][k][n]f32): [b][m][n]f32 = batched_matmul a c
entry dot [n] (a: [n]f32) (b: [n]f32): f32 = dot_product a b

-- Activation functions
entry apply_softmax [n] (x: [n]f32): [n]f32 = softmax x
entry apply_layer_norm [n] (x: [n]f32) (gamma: [n]f32) (beta: [n]f32) (eps: f32): [n]f32 = layer_norm x gamma beta eps
entry apply_relu [n] (x: [n]f32): [n]f32 = relu x
entry apply_gelu [n] (x: [n]f32): [n]f32 = gelu x

-- Optimizer operations
entry clip_fisher [n] (fisher: [n]f32) (clip_val: f32): [n]f32 = spectral_clip fisher clip_val
entry reduce_gradients [b][n] (gradients: [b][n]f32): [n]f32 = batch_reduce gradients
entry update_fisher [n] (fisher: [n]f32) (grad: [n]f32) (decay: f32): [n]f32 = fisher_diagonal_update fisher grad decay
entry compute_natural_grad [n] (grad: [n]f32) (fisher: [n]f32) (damping: f32): [n]f32 = spectral_natural_gradient grad fisher damping

-- Ranking operations
entry rank_segments [n] (query_hash: u64) (segment_hashes: [n]u64) (base_scores: [n]f32): [n]f32 = score_segments query_hash segment_hashes base_scores
entry select_topk [n] (k: i64) (scores: [n]f32): ([k]f32, [k]i64) = topk k scores (iota n)

-- RSF operations
entry rsf_forward [n] (x: [n]f32) (s_w: [n]f32) (t_w: [n]f32) (s_b: [n]f32) (t_b: [n]f32) (perm: [n]i64): [n]f32 = rsf_forward_layer x s_w t_w s_b t_b perm
entry rsf_backward [n] (grad: [n]f32) (x: [n]f32) (s_w: [n]f32) (t_w: [n]f32) (perm: [n]i64): ([n]f32, [n]f32, [n]f32, [n]f32, [n]f32) = rsf_backward_layer grad x s_w t_w perm

-- SSI operations
entry ssi_hash_tokens [m] (tokens: [m]u32): u64 = hash_sequence tokens
entry ssi_find_nearest [n][m] (tree: [n]u64) (query: [m]u32): i64 = ssi_search tree query
entry ssi_get_topk [n][m] (tree: [n]u64) (scores: [n]f32) (query: [m]u32) (k: i64): ([k]u64, [k]f32) = ssi_retrieve_topk tree scores query k
entry ssi_similarity [m] (query: [m]u32) (candidate: [m]u32): f32 = ssi_compute_similarity query candidate

-- Hashing operations
entry compute_ngram_hashes [n] (tokens: [n]u32) (ngram_size: i64): []u64 = ngram_hash tokens ngram_size
entry compute_lsh [n] (vec: [n]f32) (num_tables: i64) (seed: u64): [num_tables]u64 = lsh_hash vec num_tables seed

-- Attention mechanism
entry compute_attention [seq_len][d_model] (query: [seq_len][d_model]f32) (key: [seq_len][d_model]f32) (value: [seq_len][d_model]f32): [seq_len][d_model]f32 = attention query key value

-- Convolution operations
entry apply_conv1d [input_len][kernel_size] (input: [input_len]f32) (kernel: [kernel_size]f32): [input_len - kernel_size + 1]f32 = conv1d input kernel
entry apply_maxpool1d [input_len] (input: [input_len]f32) (pool_size: i64): [input_len / pool_size]f32 = maxpool1d input pool_size

-- Element-wise operations
entry add_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_add a b
entry mul_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_mul a b
entry div_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_div a b
entry sub_arrays [n] (a: [n]f32) (b: [n]f32): [n]f32 = elem_sub a b

-- Scalar operations
entry add_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_add a s
entry mul_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_mul a s
entry div_scalar [n] (a: [n]f32) (s: f32): [n]f32 = scalar_div a s

-- Reduction operations
entry array_sum [n] (x: [n]f32): f32 = sum x
entry array_mean [n] (x: [n]f32): f32 = mean x
entry array_max [n] (x: [n]f32): f32 = max x
entry array_min [n] (x: [n]f32): f32 = min x
