const std = @import("std");
const mem = std.mem;
const math = std.math;
const time = std.time;
const fs = std.fs;

const types = @import("core/types.zig");
const memory = @import("core/memory.zig");
const tensor_mod = @import("core/tensor.zig");
const io = @import("core/io.zig");

const rsf_mod = @import("processor/rsf.zig");
const mgt_mod = @import("tokenizer/mgt.zig");
const sfd_mod = @import("optimizer/sfd.zig");
const ssi_mod = @import("index/ssi.zig");
const ranker_mod = @import("ranker/ranker.zig");

const Tensor = tensor_mod.Tensor;
const RSF = rsf_mod.RSF;
const MGT = mgt_mod.MGT;
const SFD = sfd_mod.SFD;
const SSI = ssi_mod.SSI;
const Ranker = ranker_mod.Ranker;
const PRNG = types.PRNG;
const RankedSegment = types.RankedSegment;

const Config = struct {
    embedding_dim: usize = 128,
    rsf_layers: usize = 4,
    batch_size: usize = 16,
    num_epochs: usize = 10,
    learning_rate: f32 = 0.001,
    num_training_samples: usize = 100,
    num_validation_samples: usize = 100,
    models_dir: []u8 = undefined,
    vocab_file: ?[]u8 = null,
    dataset_path: ?[]const u8 = null,
    sample_limit: usize = 3716,
    gradient_clip_norm: f32 = 5.0,
    validation_mse_threshold: f32 = 0.1,
    validation_confidence_level: f32 = 0.95,
    sequence_length: usize = 64,
    validation_query_length: usize = 32,
    top_k: usize = 5,
    noise_level: f32 = 0.05,
    
    models_dir_owned: bool = false,
    vocab_file_owned: bool = false,
    
    pub fn deinit(self: *Config, allocator: mem.Allocator) void {
        if (self.models_dir_owned) {
            allocator.free(self.models_dir);
        }
        if (self.vocab_file_owned) {
            if (self.vocab_file) |vf| {
                allocator.free(vf);
            }
        }
    }
    
    pub fn parseArgs(allocator: mem.Allocator) !Config {
        var config = Config{
            .models_dir = undefined,
            .dataset_path = "arxiv_hungarian_dataset 2.jsonl",
            .sample_limit = 100,
        };
        var models_dir_allocated: ?[]u8 = null;
        var vocab_file_allocated: ?[]u8 = null;
        
        errdefer {
            if (models_dir_allocated) |dir| allocator.free(dir);
            if (vocab_file_allocated) |file| allocator.free(file);
        }
        
        const default_models_dir = try allocator.dupe(u8, "models");
        models_dir_allocated = default_models_dir;
        config.models_dir = default_models_dir;
        config.models_dir_owned = true;
        
        var args = try std.process.argsWithAllocator(allocator);
        defer args.deinit();
        
        _ = args.skip();
        
        while (args.next()) |arg| {
            if (mem.eql(u8, arg, "--embedding-dim")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --embedding-dim requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.embedding_dim = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --embedding-dim value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--layers")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --layers requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.rsf_layers = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --layers value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--batch-size")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --batch-size requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.batch_size = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --batch-size value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--epochs")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --epochs requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.num_epochs = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --epochs value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--lr")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --lr requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.learning_rate = std.fmt.parseFloat(f32, val) catch |err| {
                    std.debug.print("Error: Invalid --lr value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--samples")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --samples requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.num_training_samples = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --samples value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--models-dir")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --models-dir requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                if (models_dir_allocated) |old| allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                models_dir_allocated = duped;
                config.models_dir = duped;
                config.models_dir_owned = true;
            } else if (mem.eql(u8, arg, "--vocab-file")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --vocab-file requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                if (vocab_file_allocated) |old| allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                vocab_file_allocated = duped;
                config.vocab_file = duped;
                config.vocab_file_owned = true;
            } else if (mem.eql(u8, arg, "--gradient-clip")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --gradient-clip requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.gradient_clip_norm = std.fmt.parseFloat(f32, val) catch |err| {
                    std.debug.print("Error: Invalid --gradient-clip value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--sequence-length")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --sequence-length requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.sequence_length = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --sequence-length value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--top-k")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --top-k requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.top_k = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --top-k value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--noise-level")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --noise-level requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.noise_level = std.fmt.parseFloat(f32, val) catch |err| {
                    std.debug.print("Error: Invalid --noise-level value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--help")) {
                try printHelp();
                std.process.exit(0);
            }
        }
        
        if (config.num_training_samples == 0) {
            std.debug.print("Error: --samples must be > 0\n", .{});
            return error.InvalidConfig;
        }
        if (config.batch_size == 0) {
            std.debug.print("Error: --batch-size must be > 0\n", .{});
            return error.InvalidConfig;
        }
        
        return config;
    }
};

const TrainingStats = struct {
    epoch: usize,
    loss: f32,
    avg_rank_score: f32,
    samples_processed: usize,
    elapsed_ms: i64,
};

const ValidationMetrics = struct {
    mse: f32,
    rmse: f32,
    mae: f32,
    r_squared: f32,
    mean_prediction: f32,
    mean_target: f32,
    confidence_interval_lower: f32,
    confidence_interval_upper: f32,
    num_samples: usize,
};

const TrainingSample = struct {
    text: []const u8,
    tokens: []u32,
};

const TerminalColors = struct {
    enabled: bool,
    reset: []const u8,
    bold: []const u8,
    cyan: []const u8,
    green: []const u8,
    yellow: []const u8,
    magenta: []const u8,
    blue: []const u8,
    red: []const u8,
    
    fn detect() TerminalColors {
        const enabled = std.io.tty.detectConfig(std.io.getStdOut()) != .no_color;
        return if (enabled) TerminalColors{
            .enabled = true,
            .reset = "\x1b[0m",
            .bold = "\x1b[1m",
            .cyan = "\x1b[36m",
            .green = "\x1b[32m",
            .yellow = "\x1b[33m",
            .magenta = "\x1b[35m",
            .blue = "\x1b[34m",
            .red = "\x1b[31m",
        } else TerminalColors{
            .enabled = false,
            .reset = "",
            .bold = "",
            .cyan = "",
            .green = "",
            .yellow = "",
            .magenta = "",
            .blue = "",
            .red = "",
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("FATAL: Memory leak detected!\n", .{});
            std.process.exit(1);
        }
    }
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();
    const colors = TerminalColors.detect();

    try printBanner(stdout, colors);
    
    var config = Config.parseArgs(allocator) catch |err| {
        std.debug.print("Failed to parse arguments: {any}\n", .{err});
        std.process.exit(1);
    };
    defer config.deinit(allocator);
    
    try stdout.writeAll("\n[INIT] Initializing JAIDE v40 components...\n");
    
    const start_time = time.milliTimestamp();

    try stdout.writeAll("[INIT] Creating MGT tokenizer with sample vocabulary...\n");
    var mgt = try initTokenizer(allocator, config.vocab_file);
    defer mgt.deinit();
    try stdout.print("[OK] MGT initialized with vocab size: {d}\n", .{mgt.vocabSize()});

    try stdout.writeAll("[INIT] Creating SSI index...\n");
    var ssi = SSI.init(allocator);
    defer ssi.deinit();
    try stdout.writeAll("[OK] SSI index initialized\n");

    try stdout.print("[INIT] Creating Ranker (ngrams={d}, lsh_tables={d})...\n", .{10, 16});
    var ranker = try Ranker.init(allocator, 10, 16, 42);
    defer ranker.deinit();
    try stdout.writeAll("[OK] Ranker initialized\n");

    try stdout.print("[INIT] Creating RSF processor (dim={d}, layers={d})...\n", .{config.embedding_dim, config.rsf_layers});
    var rsf = try RSF.init(allocator, config.embedding_dim, config.rsf_layers);
    defer rsf.deinit();
    try stdout.writeAll("[OK] RSF processor initialized\n");

    const total_params = calculateTotalParams(config.embedding_dim, config.rsf_layers);
    try stdout.print("[INIT] Creating SFD optimizer (params={d})...\n", .{total_params});
    var optimizer = try SFD.init(allocator, total_params);
    defer optimizer.deinit();
    try stdout.writeAll("[OK] SFD optimizer initialized\n");

    try stdout.writeAll("\n[DATA] Loading training dataset...\n");
    var training_data = try loadDatasetSamples(allocator, &mgt, config);
    defer {
        for (training_data) |*sample| {
            allocator.free(sample.tokens);
            allocator.free(sample.text);
        }
        allocator.free(training_data);
    }
    try stdout.print("[OK] Loaded {d} training samples from dataset\n", .{training_data.len});

    try stdout.writeAll("[INDEX] Populating SSI index with training sequences...\n");
    var i: usize = 0;
    while (i < training_data.len) : (i += 1) {
        const sample = training_data[i];
        const is_anchor = (i % 10 == 0);
        try ssi.addSequence(sample.tokens, @as(u64, @intCast(i)), is_anchor);
    }
    const ssi_stats = ssi.stats();
    try stdout.print("[OK] SSI indexed: {d} nodes, {d} leaves, depth {d}\n", .{ssi_stats.nodes, ssi_stats.leaves, ssi_stats.depth});

    try stdout.writeAll("\n========================================\n");
    try stdout.writeAll("TRAINING LOOP STARTING\n");
    try stdout.writeAll("========================================\n\n");

    var epoch: usize = 0;
    while (epoch < config.num_epochs) : (epoch += 1) {
        const epoch_start = time.milliTimestamp();
        var epoch_loss: f64 = 0.0;
        var epoch_rank_score: f64 = 0.0;
        var samples_processed: usize = 0;

        try stdout.print("\n--- Epoch {d}/{d} ---\n", .{epoch + 1, config.num_epochs});

        var batch_idx: usize = 0;
        while (batch_idx < training_data.len) {
            const batch_end = @min(batch_idx + config.batch_size, training_data.len);
            const current_batch = training_data[batch_idx..batch_end];
            const actual_batch_size = current_batch.len;

            var batch_embeddings = std.ArrayList(Tensor).init(allocator);
            defer {
                for (batch_embeddings.items) |*emb| emb.deinit();
                batch_embeddings.deinit();
            }
            
            var batch_targets = std.ArrayList(Tensor).init(allocator);
            defer {
                for (batch_targets.items) |*tgt| tgt.deinit();
                batch_targets.deinit();
            }
            
            var batch_rank_scores = std.ArrayList(f64).init(allocator);
            defer batch_rank_scores.deinit();

            var sample_idx: usize = 0;
            while (sample_idx < actual_batch_size) : (sample_idx += 1) {
                const sample = current_batch[sample_idx];
                const query_len = @min(config.sequence_length, sample.tokens.len);
                const query_tokens = sample.tokens[0..query_len];
                
                var top_k = try ssi.retrieveTopK(query_tokens, config.top_k, allocator);
                defer {
                    for (top_k) |*seg| seg.deinit(allocator);
                    allocator.free(top_k);
                }

                try ranker.rankCandidates(top_k, &ssi, allocator);
                
                var rank_sum: f64 = 0.0;
                for (top_k) |seg| rank_sum += seg.score;
                const avg_rank = if (top_k.len > 0) rank_sum / @as(f64, @floatFromInt(top_k.len)) else 0.0;
                try batch_rank_scores.append(avg_rank);

                const input_len = @min(sample.tokens.len, config.sequence_length);
                var input_embedding = try createEmbedding(allocator, sample.tokens[0..input_len], config.embedding_dim, mgt.vocabSize());
                try batch_embeddings.append(input_embedding);

                const next_idx = batch_idx + sample_idx + 1;
                var target_embedding = if (next_idx < training_data.len) blk: {
                    const target_len = @min(training_data[next_idx].tokens.len, config.sequence_length);
                    const target_tokens = training_data[next_idx].tokens[0..target_len];
                    break :blk try createEmbedding(allocator, target_tokens, config.embedding_dim, mgt.vocabSize());
                } else blk: {
                    var t = try input_embedding.copy(allocator);
                    addNoise(&t, config.noise_level);
                    break :blk t;
                };
                try batch_targets.append(target_embedding);
            }

            var batch_loss: f64 = 0.0;
            
            var batch_gradients = try Tensor.init(allocator, &.{total_params});
            defer batch_gradients.deinit();
            
            for (batch_gradients.data) |*val| val.* = 0.0;

            var emb_idx: usize = 0;
            while (emb_idx < batch_embeddings.items.len) : (emb_idx += 1) {
                var input_backup = try batch_embeddings.items[emb_idx].copy(allocator);
                defer input_backup.deinit();
                
                var rsf_output = try batch_embeddings.items[emb_idx].copy(allocator);
                defer rsf_output.deinit();
                try rsf.forward(&rsf_output);

                const loss = try computeMSELoss(&rsf_output, &batch_targets.items[emb_idx]);
                batch_loss += loss;

                var grad_output = try rsf_output.copy(allocator);
                defer grad_output.deinit();
                
                try grad_output.sub(&batch_targets.items[emb_idx]);
                const scale = 2.0 / @as(f32, @floatFromInt(rsf_output.data.len));
                grad_output.mulScalar(scale);

                var grad_input = try rsf.backward(&grad_output, &input_backup);
                defer grad_input.deinit();

                var sample_gradients = try extractGradients(allocator, &rsf, total_params);
                defer sample_gradients.deinit();
                
                var g_idx: usize = 0;
                while (g_idx < total_params) : (g_idx += 1) {
                    batch_gradients.data[g_idx] += sample_gradients.data[g_idx];
                }
            }

            const batch_scale = 1.0 / @as(f32, @floatFromInt(actual_batch_size));
            for (batch_gradients.data) |*val| val.* *= batch_scale;
            
            clipGradients(&batch_gradients, config.gradient_clip_norm);
            
            var params = try extractParameters(allocator, &rsf, total_params);
            defer params.deinit();
            
            try optimizer.update(&batch_gradients, &params, config.learning_rate);
            
            try updateRSFParameters(&rsf, &params);

            epoch_loss += batch_loss;
            for (batch_rank_scores.items) |score| epoch_rank_score += score;
            samples_processed += actual_batch_size;

            if ((batch_idx / config.batch_size) % 5 == 0) {
                const avg_loss = if (samples_processed > 0) epoch_loss / @as(f64, @floatFromInt(samples_processed)) else 0.0;
                const num_batches = (training_data.len + config.batch_size - 1) / config.batch_size;
                const current_batch_num = batch_idx / config.batch_size + 1;
                try stdout.print("  Batch {d}/{d}: loss={d:.6} samples={d}\n", 
                    .{current_batch_num, num_batches, avg_loss, samples_processed});
            }
            
            batch_idx = batch_end;
        }

        const epoch_time = time.milliTimestamp() - epoch_start;
        const stats = TrainingStats{
            .epoch = epoch + 1,
            .loss = if (samples_processed > 0) @as(f32, @floatCast(epoch_loss / @as(f64, @floatFromInt(samples_processed)))) else 0.0,
            .avg_rank_score = if (samples_processed > 0) @as(f32, @floatCast(epoch_rank_score / @as(f64, @floatFromInt(samples_processed)))) else 0.0,
            .samples_processed = samples_processed,
            .elapsed_ms = epoch_time,
        };

        try printEpochStats(stdout, stats, colors);
    }

    try stdout.writeAll("\n[SAVE] Saving trained models...\n");
    
    try ensureDirectoryExists(config.models_dir);
    
    const rsf_path = try std.fmt.allocPrint(allocator, "{s}/rsf_trained.bin", .{config.models_dir});
    defer allocator.free(rsf_path);
    try rsf.save(rsf_path);
    try stdout.print("[OK] RSF model saved to {s}\n", .{rsf_path});
    
    const mgt_path = try std.fmt.allocPrint(allocator, "{s}/mgt_vocab.bin", .{config.models_dir});
    defer allocator.free(mgt_path);
    try mgt.saveVocab(mgt_path);
    try stdout.print("[OK] MGT vocabulary saved to {s}\n", .{mgt_path});
    
    const opt_path = try std.fmt.allocPrint(allocator, "{s}/optimizer_state.bin", .{config.models_dir});
    defer allocator.free(opt_path);
    try optimizer.saveState(opt_path);
    try stdout.print("[OK] Optimizer state saved to {s}\n", .{opt_path});
    
    const ranker_path = try std.fmt.allocPrint(allocator, "{s}/ranker_weights.bin", .{config.models_dir});
    defer allocator.free(ranker_path);
    try ranker.exportModel(ranker_path);
    try stdout.print("[OK] Ranker weights saved to {s}\n", .{ranker_path});

    try stdout.writeAll("\n[VALIDATE] Running final validation...\n");
    const metrics = try runValidation(allocator, &rsf, &mgt, &ssi, &ranker, config);
    try printValidationMetrics(stdout, metrics, colors);

    const total_time = time.milliTimestamp() - start_time;
    try stdout.writeAll("\n========================================\n");
    try stdout.writeAll("TRAINING COMPLETED\n");
    try stdout.writeAll("========================================\n");
    try stdout.print("Total time: {d}ms ({d:.2}s)\n", .{total_time, @as(f64, @floatFromInt(total_time)) / 1000.0});
    try stdout.print("Total samples processed: {d}\n", .{config.num_training_samples * config.num_epochs});
    try stdout.print("Final validation MSE: {d:.8}\n", .{metrics.mse});
    try stdout.print("Final validation RMSE: {d:.8}\n", .{metrics.rmse});
    try stdout.print("Final validation R² Score: {d:.6}\n", .{metrics.r_squared});
    try stdout.print("\nModels saved to {s}/\n", .{config.models_dir});
    try stdout.writeAll("========================================\n");
    
    if (config.num_epochs > 0) {
        try runIntegrationTests(allocator, &rsf, &mgt, &ssi, &ranker, config, colors);
    }
}

fn printBanner(writer: anytype, colors: TerminalColors) !void {
    try writer.print("{s}{s}========================================{s}\n", .{colors.bold, colors.cyan, colors.reset});
    try writer.print("{s}{s}JAIDE v40 - Root-Level LLM System{s}\n", .{colors.bold, colors.cyan, colors.reset});
    try writer.print("{s}{s}========================================{s}\n", .{colors.bold, colors.cyan, colors.reset});
    try writer.writeAll("\n");
    try writer.print("{s}Architecture:{s} Jade Neural (Non-Transformer)\n", .{colors.bold, colors.reset});
    try writer.print("{s}Context:{s} 50M+ tokens via SSI\n", .{colors.bold, colors.reset});
    try writer.print("{s}Components:{s}\n", .{colors.bold, colors.reset});
    try writer.print("  {s}•{s} {s}SSI:{s} Succinct Semantic Index\n", .{colors.green, colors.reset, colors.bold, colors.reset});
    try writer.print("  {s}•{s} {s}Ranker:{s} Non-attention relevance scoring\n", .{colors.green, colors.reset, colors.bold, colors.reset});
    try writer.print("  {s}•{s} {s}RSF:{s} Reversible Scatter-Flow processor\n", .{colors.green, colors.reset, colors.bold, colors.reset});
    try writer.print("  {s}•{s} {s}MGT:{s} Morpho-Graph Tokenizer\n", .{colors.green, colors.reset, colors.bold, colors.reset});
    try writer.print("  {s}•{s} {s}SFD:{s} Spectral Fisher Diagonalizer\n", .{colors.green, colors.reset, colors.bold, colors.reset});
    try writer.writeAll("\n");
    try writer.print("{s}Formal Guarantees:{s} {s}Lean, Isabelle, Agda, Viper, TLA+, Spin{s}\n", .{colors.yellow, colors.reset, colors.magenta, colors.reset});
    try writer.print("{s}Hardware:{s} {s}Clash RTL + Futhark kernels{s}\n", .{colors.yellow, colors.reset, colors.blue, colors.reset});
    try writer.print("{s}ZK Proofs:{s} {s}Circom inference verification{s}\n", .{colors.yellow, colors.reset, colors.blue, colors.reset});
    try writer.writeAll("\n");
}

fn printHelp() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("JAIDE v40 - Usage:\n\n");
    try stdout.writeAll("Options:\n");
    try stdout.writeAll("  --embedding-dim <n>      Embedding dimension (default: 128)\n");
    try stdout.writeAll("  --layers <n>             Number of RSF layers (default: 4)\n");
    try stdout.writeAll("  --batch-size <n>         Batch size (default: 16)\n");
    try stdout.writeAll("  --epochs <n>             Number of epochs (default: 10)\n");
    try stdout.writeAll("  --lr <f>                 Learning rate (default: 0.001)\n");
    try stdout.writeAll("  --samples <n>            Training samples (default: 100)\n");
    try stdout.writeAll("  --models-dir <path>      Models directory (default: models)\n");
    try stdout.writeAll("  --vocab-file <path>      Load vocabulary from file (optional)\n");
    try stdout.writeAll("  --gradient-clip <f>      Gradient clipping norm (default: 5.0)\n");
    try stdout.writeAll("  --sequence-length <n>    Sequence length for queries (default: 64)\n");
    try stdout.writeAll("  --top-k <n>              Number of top candidates to retrieve (default: 5)\n");
    try stdout.writeAll("  --noise-level <f>        Noise level for augmentation (default: 0.05)\n");
    try stdout.writeAll("  --help                   Show this help message\n");
}

fn initTokenizer(allocator: mem.Allocator, vocab_file: ?[]const u8) !MGT {
    if (vocab_file) |file_path| {
        const file = fs.cwd().openFile(file_path, .{}) catch |err| {
            std.debug.print("Warning: Could not load vocab file {s}: {any}\n", .{file_path, err});
            std.debug.print("Falling back to default vocabulary\n", .{});
            return try initDefaultTokenizer(allocator);
        };
        defer file.close();
        
        var vocab_list = std.ArrayList([]const u8).init(allocator);
        defer vocab_list.deinit();
        
        var buf_reader = std.io.bufferedReader(file.reader());
        var in_stream = buf_reader.reader();
        
        var buf: [1024]u8 = undefined;
        while (try in_stream.readUntilDelimiterOrEof(&buf, '\n')) |line| {
            const trimmed = mem.trim(u8, line, &std.ascii.whitespace);
            if (trimmed.len > 0) {
                const word = try allocator.dupe(u8, trimmed);
                try vocab_list.append(word);
            }
        }
        
        if (vocab_list.items.len == 0) {
            std.debug.print("Warning: Vocab file is empty, using default vocabulary\n", .{});
            return try initDefaultTokenizer(allocator);
        }
        
        const anchors = &[_][]const u8{"neural", "network", "model", "train", "learning"};
        return try MGT.init(allocator, vocab_list.items, anchors);
    } else {
        return try initDefaultTokenizer(allocator);
    }
}

fn initDefaultTokenizer(allocator: mem.Allocator) !MGT {
    const vocab = &[_][]const u8{
        "the", "is", "at", "which", "on", "a", "an", "as", "are", "was",
        "be", "by", "for", "from", "has", "he", "in", "it", "its", "of",
        "that", "to", "with", "and", "or", "not", "but", "can", "will", "would",
        "could", "should", "have", "had", "does", "did", "make", "made", "get", "got",
        "go", "come", "take", "see", "know", "think", "look", "want", "give", "gave",
        "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call",
        "neural", "network", "model", "train", "learn", "data", "input", "output",
        "layer", "weight", "bias", "gradient", "loss", "optimizer", "epoch", "batch",
        "tensor", "matrix", "vector", "scalar", "forward", "backward", "parameter",
        "learning", "rate", "decay", "momentum", "regularization", "dropout",
        "activation", "function", "sigmoid", "relu", "tanh", "softmax", "embedding",
        "processing", "optimization", "convergence", "iteration", "computation",
    };
    
    const anchors = &[_][]const u8{"neural", "network", "model", "train", "learning"};
    
    return try MGT.init(allocator, vocab, anchors);
}

fn loadDatasetSamples(allocator: mem.Allocator, mgt: *MGT, config: Config) ![]TrainingSample {
    const dataset_file = config.dataset_path orelse return error.DatasetPathNotSpecified;
    
    std.debug.print("[DEBUG] Opening dataset file: {s}\n", .{dataset_file});
    const file = fs.cwd().openFile(dataset_file, .{}) catch |err| {
        std.debug.print("[ERROR] Failed to open dataset file '{s}': {}\n", .{dataset_file, err});
        return err;
    };
    defer file.close();
    std.debug.print("[DEBUG] File opened successfully\n", .{});
    
    var buf_reader = std.io.bufferedReader(file.reader());
    const reader = buf_reader.reader();
    
    var samples_list = std.ArrayList(TrainingSample).init(allocator);
    errdefer {
        for (samples_list.items) |*sample| {
            allocator.free(sample.tokens);
            allocator.free(sample.text);
        }
        samples_list.deinit();
    }
    
    var line_buf = std.ArrayList(u8).init(allocator);
    defer line_buf.deinit();
    
    var line_num: usize = 0;
    while (true) : (line_num += 1) {
        if (samples_list.items.len >= config.sample_limit) break;
        
        if (line_num % 10 == 0 and line_num > 0) {
            std.debug.print("[DEBUG] Processing line {d}, samples loaded: {d}\n", .{line_num, samples_list.items.len});
        }
        
        line_buf.clearRetainingCapacity();
        reader.streamUntilDelimiter(line_buf.writer(), '\n', null) catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };
        
        if (line_buf.items.len == 0) continue;
        
        var parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            line_buf.items,
            .{},
        ) catch |err| {
            std.debug.print("[WARN] Line {d}: Malformed JSON, skipping: {}\n", .{line_num + 1, err});
            continue;
        };
        defer parsed.deinit();
        
        const obj = parsed.value.object;
        const instruction = obj.get("instruction") orelse {
            std.debug.print("[WARN] Line {d}: Missing 'instruction' field, skipping\n", .{line_num + 1});
            continue;
        };
        const input = obj.get("input") orelse {
            std.debug.print("[WARN] Line {d}: Missing 'input' field, skipping\n", .{line_num + 1});
            continue;
        };
        const output = obj.get("output") orelse {
            std.debug.print("[WARN] Line {d}: Missing 'output' field, skipping\n", .{line_num + 1});
            continue;
        };
        
        var text_buf = std.ArrayList(u8).init(allocator);
        errdefer text_buf.deinit();
        
        try text_buf.appendSlice(instruction.string);
        if (input.string.len > 0) {
            try text_buf.append(' ');
            try text_buf.appendSlice(input.string);
        }
        if (output.string.len > 0) {
            try text_buf.append(' ');
            try text_buf.appendSlice(output.string);
        }
        
        const text = try text_buf.toOwnedSlice();
        errdefer allocator.free(text);
        
        if (samples_list.items.len < 3) {
            std.debug.print("[DEBUG] Sample {d} text length: {d} chars\n", .{samples_list.items.len + 1, text.len});
        }
        
        var token_list = std.ArrayList(u32).init(allocator);
        defer token_list.deinit();
        
        if (samples_list.items.len < 3) {
            std.debug.print("[DEBUG] Starting tokenization for sample {d}...\n", .{samples_list.items.len + 1});
        }
        try mgt.encode(text, &token_list);
        if (samples_list.items.len < 3) {
            std.debug.print("[DEBUG] Tokenization complete, tokens: {d}\n", .{token_list.items.len});
        }
        
        const tokens = try token_list.toOwnedSlice();
        errdefer allocator.free(tokens);
        
        try samples_list.append(TrainingSample{
            .text = text,
            .tokens = tokens,
        });
    }
    
    std.debug.print("[DEBUG] Dataset loading complete. Total samples: {d}\n", .{samples_list.items.len});
    
    if (samples_list.items.len == 0) {
        return error.NoValidSamples;
    }
    
    return samples_list.toOwnedSlice();
}

fn createEmbedding(allocator: mem.Allocator, tokens: []const u32, dim: usize, vocab_size: usize) !Tensor {
    var embedding = try Tensor.init(allocator, &.{1, dim * 2});
    errdefer embedding.deinit();
    
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        if (i < tokens.len) {
            const token = tokens[i];
            const normalized = @as(f32, @floatFromInt(token % vocab_size)) / @as(f32, @floatFromInt(vocab_size));
            embedding.data[i] = normalized;
            embedding.data[i + dim] = normalized;
        } else {
            embedding.data[i] = 0.0;
            embedding.data[i + dim] = 0.0;
        }
    }
    
    return embedding;
}

fn addNoise(tensor: *Tensor, scale: f32) void {
    var prng = PRNG.init(@as(u64, @bitCast(time.milliTimestamp())));
    for (tensor.data) |*val| {
        const noise = (prng.float() - 0.5) * 2.0 * scale;
        const clamped = @max(-scale, @min(scale, noise));
        val.* += clamped;
    }
}

fn computeMSELoss(output: *const Tensor, target: *const Tensor) !f32 {
    if (output.data.len != target.data.len) return error.ShapeMismatch;
    
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < output.data.len) : (i += 1) {
        const diff = @as(f64, output.data[i]) - @as(f64, target.data[i]);
        sum += diff * diff;
    }
    
    const count = @as(f64, @floatFromInt(output.data.len));
    return if (count > 0) @as(f32, @floatCast(sum / count)) else 0.0;
}

fn calculateTotalParams(dim: usize, num_layers: usize) usize {
    const s_weight_size = dim * dim;
    const t_weight_size = dim * dim;
    const s_bias_size = dim;
    const t_bias_size = dim;
    const params_per_layer = s_weight_size + t_weight_size + s_bias_size + t_bias_size;
    return params_per_layer * num_layers;
}

fn extractGradients(allocator: mem.Allocator, rsf: *const RSF, total_params: usize) !Tensor {
    var grads = try Tensor.init(allocator, &.{total_params});
    errdefer grads.deinit();
    
    var offset: usize = 0;
    for (rsf.layers) |layer| {
        const s_weight_size = layer.s_weight_grad.data.len;
        const t_weight_size = layer.t_weight_grad.data.len;
        const s_bias_size = layer.s_bias_grad.data.len;
        const t_bias_size = layer.t_bias_grad.data.len;
        
        if (offset + s_weight_size <= total_params) {
            var i: usize = 0;
            while (i < s_weight_size) : (i += 1) {
                grads.data[offset + i] = layer.s_weight_grad.data[i];
            }
            offset += s_weight_size;
        }
        
        if (offset + t_weight_size <= total_params) {
            var i: usize = 0;
            while (i < t_weight_size) : (i += 1) {
                grads.data[offset + i] = layer.t_weight_grad.data[i];
            }
            offset += t_weight_size;
        }
        
        if (offset + s_bias_size <= total_params) {
            var i: usize = 0;
            while (i < s_bias_size) : (i += 1) {
                grads.data[offset + i] = layer.s_bias_grad.data[i];
            }
            offset += s_bias_size;
        }
        
        if (offset + t_bias_size <= total_params) {
            var i: usize = 0;
            while (i < t_bias_size) : (i += 1) {
                grads.data[offset + i] = layer.t_bias_grad.data[i];
            }
            offset += t_bias_size;
        }
    }
    
    return grads;
}

fn extractParameters(allocator: mem.Allocator, rsf: *const RSF, total_params: usize) !Tensor {
    var params = try Tensor.init(allocator, &.{total_params});
    errdefer params.deinit();
    
    var offset: usize = 0;
    for (rsf.layers) |layer| {
        const s_weight_size = layer.s_weight.data.len;
        const t_weight_size = layer.t_weight.data.len;
        const s_bias_size = layer.s_bias.data.len;
        const t_bias_size = layer.t_bias.data.len;
        
        if (offset + s_weight_size <= total_params) {
            var i: usize = 0;
            while (i < s_weight_size) : (i += 1) {
                params.data[offset + i] = layer.s_weight.data[i];
            }
            offset += s_weight_size;
        }
        
        if (offset + t_weight_size <= total_params) {
            var i: usize = 0;
            while (i < t_weight_size) : (i += 1) {
                params.data[offset + i] = layer.t_weight.data[i];
            }
            offset += t_weight_size;
        }
        
        if (offset + s_bias_size <= total_params) {
            var i: usize = 0;
            while (i < s_bias_size) : (i += 1) {
                params.data[offset + i] = layer.s_bias.data[i];
            }
            offset += s_bias_size;
        }
        
        if (offset + t_bias_size <= total_params) {
            var i: usize = 0;
            while (i < t_bias_size) : (i += 1) {
                params.data[offset + i] = layer.t_bias.data[i];
            }
            offset += t_bias_size;
        }
    }
    
    return params;
}

fn updateRSFParameters(rsf: *RSF, params: *const Tensor) !void {
    var offset: usize = 0;
    for (rsf.layers) |*layer| {
        const s_weight_size = layer.s_weight.data.len;
        const t_weight_size = layer.t_weight.data.len;
        const s_bias_size = layer.s_bias.data.len;
        const t_bias_size = layer.t_bias.data.len;
        
        if (offset + s_weight_size <= params.data.len) {
            var i: usize = 0;
            while (i < s_weight_size) : (i += 1) {
                layer.s_weight.data[i] = params.data[offset + i];
            }
            offset += s_weight_size;
        }
        
        if (offset + t_weight_size <= params.data.len) {
            var i: usize = 0;
            while (i < t_weight_size) : (i += 1) {
                layer.t_weight.data[i] = params.data[offset + i];
            }
            offset += t_weight_size;
        }
        
        if (offset + s_bias_size <= params.data.len) {
            var i: usize = 0;
            while (i < s_bias_size) : (i += 1) {
                layer.s_bias.data[i] = params.data[offset + i];
            }
            offset += s_bias_size;
        }
        
        if (offset + t_bias_size <= params.data.len) {
            var i: usize = 0;
            while (i < t_bias_size) : (i += 1) {
                layer.t_bias.data[i] = params.data[offset + i];
            }
            offset += t_bias_size;
        }
    }
}

fn clipGradients(gradients: *Tensor, max_norm: f32) void {
    var norm_sq: f64 = 0.0;
    for (gradients.data) |val| {
        const v = @as(f64, val);
        norm_sq += v * v;
    }
    const norm = @sqrt(norm_sq);
    
    if (norm > max_norm) {
        const scale = @as(f32, @floatCast(@as(f64, max_norm) / norm));
        for (gradients.data) |*val| {
            val.* *= scale;
        }
    }
}

fn runValidation(allocator: mem.Allocator, rsf: *RSF, mgt: *MGT, ssi: *SSI, ranker: *const Ranker, config: Config) !ValidationMetrics {
    const timestamp = time.milliTimestamp();
    const seed = if (timestamp >= 0) @as(u64, @intCast(timestamp)) else @as(u64, @intCast(-timestamp));
    var prng = PRNG.init(seed);
    
    const val_templates = [_][]const u8{
        "neural network training with gradient descent optimization",
        "the model learns from data to make accurate predictions",
        "deep learning requires large datasets and computational resources",
        "backpropagation computes gradients for weight updates",
        "regularization prevents overfitting on training data",
        "embedding vectors represent discrete tokens in continuous space",
        "attention mechanisms focus on relevant input features",
        "loss functions measure prediction error during training",
        "optimization algorithms update model parameters iteratively",
        "validation metrics evaluate model performance on unseen data",
        "convolutional layers extract spatial hierarchies",
        "recurrent networks model temporal dependencies",
        "batch normalization accelerates convergence",
        "dropout reduces model overfitting",
        "learning rate schedules improve optimization",
        "cross-entropy loss for classification tasks",
        "mean squared error for regression problems",
        "stochastic gradient descent with momentum",
        "adaptive learning rate methods like Adam",
        "weight initialization affects training stability",
    };
    
    const num_val_samples = config.num_validation_samples;
    
    var val_texts = try allocator.alloc([]const u8, num_val_samples);
    defer allocator.free(val_texts);
    
    var used_indices = std.AutoHashMap(usize, void).init(allocator);
    defer used_indices.deinit();
    
    var sample_idx: usize = 0;
    while (sample_idx < val_texts.len) : (sample_idx += 1) {
        var idx = prng.uniform(0, val_templates.len);
        var attempts: usize = 0;
        while (used_indices.contains(idx) and attempts < 50) : (attempts += 1) {
            idx = prng.uniform(0, val_templates.len);
        }
        try used_indices.put(idx, {});
        val_texts[sample_idx] = val_templates[idx % val_templates.len];
    }
    
    var squared_errors = try allocator.alloc(f32, num_val_samples);
    defer allocator.free(squared_errors);
    var absolute_errors = try allocator.alloc(f32, num_val_samples);
    defer allocator.free(absolute_errors);
    var predictions = try allocator.alloc(f32, num_val_samples);
    defer allocator.free(predictions);
    var targets = try allocator.alloc(f32, num_val_samples);
    defer allocator.free(targets);
    
    var total_se: f64 = 0.0;
    var total_ae: f64 = 0.0;
    var sum_predictions: f64 = 0.0;
    var sum_targets: f64 = 0.0;
    
    var idx: usize = 0;
    while (idx < num_val_samples) : (idx += 1) {
        const val_text = val_texts[idx];
        var token_list = std.ArrayList(u32).init(allocator);
        defer token_list.deinit();
        try mgt.encode(val_text, &token_list);
        const tokens = try token_list.toOwnedSlice();
        defer allocator.free(tokens);
        
        const query_len = @min(tokens.len, config.validation_query_length);
        const input_len = @min(tokens.len, config.embedding_dim);
        var input = try createEmbedding(allocator, tokens[0..input_len], config.embedding_dim, mgt.vocabSize());
        defer input.deinit();
        
        var input_backup = try input.copy(allocator);
        defer input_backup.deinit();
        
        var output = try input.copy(allocator);
        defer output.deinit();
        try rsf.forward(&output);
        
        var top_k = try ssi.retrieveTopK(tokens[0..query_len], config.top_k, allocator);
        defer {
            for (top_k) |*seg| seg.deinit(allocator);
            allocator.free(top_k);
        }
        
        try ranker.rankCandidates(top_k, ssi, allocator);
        
        const next_idx = (idx + 1) % num_val_samples;
        var next_token_list = std.ArrayList(u32).init(allocator);
        defer next_token_list.deinit();
        try mgt.encode(val_texts[next_idx], &next_token_list);
        const next_tokens = try next_token_list.toOwnedSlice();
        defer allocator.free(next_tokens);
        
        const target_len = @min(next_tokens.len, config.embedding_dim);
        var target = try createEmbedding(allocator, next_tokens[0..target_len], config.embedding_dim, mgt.vocabSize());
        defer target.deinit();
        
        var sample_se: f64 = 0.0;
        var sample_ae: f64 = 0.0;
        var sample_pred_mean: f64 = 0.0;
        var sample_target_mean: f64 = 0.0;
        
        const cmp_len = @min(output.data.len, target.data.len);
        var i: usize = 0;
        while (i < cmp_len) : (i += 1) {
            const pred = @as(f64, output.data[i]);
            const targ = @as(f64, target.data[i]);
            const diff = pred - targ;
            sample_se += diff * diff;
            sample_ae += @fabs(diff);
            sample_pred_mean += pred;
            sample_target_mean += targ;
        }
        
        const sample_mse = @as(f32, @floatCast(sample_se / @as(f64, @floatFromInt(cmp_len))));
        const sample_mae = @as(f32, @floatCast(sample_ae / @as(f64, @floatFromInt(cmp_len))));
        
        squared_errors[idx] = sample_mse;
        absolute_errors[idx] = sample_mae;
        predictions[idx] = @as(f32, @floatCast(sample_pred_mean / @as(f64, @floatFromInt(cmp_len))));
        targets[idx] = @as(f32, @floatCast(sample_target_mean / @as(f64, @floatFromInt(cmp_len))));
        
        total_se += sample_se;
        total_ae += sample_ae;
        sum_predictions += sample_pred_mean;
        sum_targets += sample_target_mean;
    }
    
    const total_predictions = @as(f64, @floatFromInt(num_val_samples * config.embedding_dim * 2));
    const mse = @as(f32, @floatCast(total_se / total_predictions));
    const rmse = @sqrt(mse);
    const mae = @as(f32, @floatCast(total_ae / total_predictions));
    const mean_prediction = @as(f32, @floatCast(sum_predictions / total_predictions));
    const mean_target = @as(f32, @floatCast(sum_targets / total_predictions));
    
    var ss_tot: f64 = 0.0;
    var ss_res: f64 = 0.0;
    idx = 0;
    while (idx < num_val_samples) : (idx += 1) {
        const targ_dev = @as(f64, targets[idx]) - @as(f64, mean_target);
        const residual = @as(f64, predictions[idx]) - @as(f64, targets[idx]);
        ss_tot += targ_dev * targ_dev;
        ss_res += residual * residual;
    }
    
    const r_squared = if (ss_tot > 0.0) 
        @as(f32, @floatCast(1.0 - (ss_res / ss_tot)))
    else
        0.0;
    
    var variance: f64 = 0.0;
    idx = 0;
    while (idx < num_val_samples) : (idx += 1) {
        const dev = @as(f64, squared_errors[idx]) - @as(f64, mse);
        variance += dev * dev;
    }
    variance /= @as(f64, @floatFromInt(num_val_samples - 1));
    const std_dev = @sqrt(variance);
    
    const z_score = 1.96;
    const se = std_dev / @sqrt(@as(f64, @floatFromInt(num_val_samples)));
    const ci_lower = @as(f32, @floatCast(@as(f64, mse) - z_score * se));
    const ci_upper = @as(f32, @floatCast(@as(f64, mse) + z_score * se));
    
    return ValidationMetrics{
        .mse = mse,
        .rmse = rmse,
        .mae = mae,
        .r_squared = r_squared,
        .mean_prediction = mean_prediction,
        .mean_target = mean_target,
        .confidence_interval_lower = @max(0.0, ci_lower),
        .confidence_interval_upper = ci_upper,
        .num_samples = num_val_samples,
    };
}

fn printEpochStats(writer: anytype, stats: TrainingStats, colors: TerminalColors) !void {
    try writer.print("{s}[EPOCH {d} COMPLETE]{s}\n", .{colors.bold, stats.epoch, colors.reset});
    try writer.print("  Loss: {s}{d:.6}{s}\n", .{colors.cyan, stats.loss, colors.reset});
    try writer.print("  Avg Rank Score: {s}{d:.4}{s}\n", .{colors.cyan, stats.avg_rank_score, colors.reset});
    try writer.print("  Samples Processed: {s}{d}{s}\n", .{colors.cyan, stats.samples_processed, colors.reset});
    try writer.print("  Time: {s}{d}ms{s}\n", .{colors.cyan, stats.elapsed_ms, colors.reset});
}

fn printValidationMetrics(writer: anytype, metrics: ValidationMetrics, colors: TerminalColors) !void {
    try writer.print("{s}Validation Metrics (n={d}):{s}\n", .{colors.bold, metrics.num_samples, colors.reset});
    try writer.print("  MSE: {s}{d:.8}{s}\n", .{colors.cyan, metrics.mse, colors.reset});
    try writer.print("  RMSE: {s}{d:.8}{s}\n", .{colors.cyan, metrics.rmse, colors.reset});
    try writer.print("  MAE: {s}{d:.8}{s}\n", .{colors.cyan, metrics.mae, colors.reset});
    try writer.print("  R² Score: {s}{d:.6}{s}\n", .{colors.green, metrics.r_squared, colors.reset});
    try writer.print("  Mean Prediction: {s}{d:.6}{s}\n", .{colors.yellow, metrics.mean_prediction, colors.reset});
    try writer.print("  Mean Target: {s}{d:.6}{s}\n", .{colors.yellow, metrics.mean_target, colors.reset});
    try writer.print("  95% CI: {s}[{d:.8}, {d:.8}]{s}\n", .{colors.magenta, metrics.confidence_interval_lower, metrics.confidence_interval_upper, colors.reset});
}

fn ensureDirectoryExists(path: []const u8) !void {
    fs.cwd().makePath(path) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
}

fn runIntegrationTests(allocator: mem.Allocator, rsf: *RSF, mgt: *MGT, ssi: *SSI, ranker: *const Ranker, config: Config, colors: TerminalColors) !void {
    const stdout = std.io.getStdOut().writer();
    
    try stdout.writeAll("\n========================================\n");
    try stdout.writeAll("INTEGRATION TESTS\n");
    try stdout.writeAll("========================================\n\n");
    
    var passed_tests: usize = 0;
    var failed_tests: usize = 0;
    const total_tests: usize = 6;
    
    try stdout.print("{s}[TEST 1] RSF Forward/Backward Pass with Gradient Verification{s}\n", .{colors.bold, colors.reset});
    {
        var test_input = try Tensor.init(allocator, &.{1, config.embedding_dim * 2});
        defer test_input.deinit();
        test_input.fill(0.5);
        
        var test_input_copy = try test_input.copy(allocator);
        defer test_input_copy.deinit();
        
        var forward_output = try test_input.copy(allocator);
        defer forward_output.deinit();
        try rsf.forward(&forward_output);
        
        var output_changed = false;
        var i: usize = 0;
        while (i < forward_output.data.len) : (i += 1) {
            if (math.fabs(forward_output.data[i] - 0.5) > 1e-6) {
                output_changed = true;
                break;
            }
        }
        
        var grad_out = try forward_output.copy(allocator);
        defer grad_out.deinit();
        grad_out.fill(0.1);
        
        var grad_in = try rsf.backward(&grad_out, &test_input_copy);
        defer grad_in.deinit();
        
        const has_valid_output = test_input.data.len > 0 and grad_in.data.len > 0 and output_changed;
        
        var grad_is_nonzero = false;
        i = 0;
        while (i < grad_in.data.len) : (i += 1) {
            if (math.fabs(grad_in.data[i]) > 1e-9) {
                grad_is_nonzero = true;
                break;
            }
        }
        
        const test_passed = has_valid_output and grad_is_nonzero;
        if (test_passed) passed_tests += 1 else failed_tests += 1;
        
        try stdout.print("  Forward changed output: {s}\n", .{if (output_changed) "YES" else "NO"});
        try stdout.print("  Backward gradients nonzero: {s}\n", .{if (grad_is_nonzero) "YES" else "NO"});
        try stdout.print("  Result: {s}{s}{s}\n\n", .{
            if (test_passed) colors.green else colors.red,
            if (test_passed) "PASSED" else "FAILED",
            colors.reset,
        });
    }
    
    try stdout.print("{s}[TEST 2] MGT Encoding/Decoding Quality Verification{s}\n", .{colors.bold, colors.reset});
    {
        const test_text = "neural network training optimization";
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        try mgt.encode(test_text, &tokens);
        
        var decoded_text = std.ArrayList(u8).init(allocator);
        defer decoded_text.deinit();
        try mgt.decode(tokens.items, &decoded_text);
        
        const has_tokens = tokens.items.len > 0 and tokens.items.len <= 100;
        const has_decoded = decoded_text.items.len > 0;
        
        var valid_token_range = true;
        const vocab_size = mgt.vocabSize();
        for (tokens.items) |token| {
            if (token >= vocab_size) {
                valid_token_range = false;
                break;
            }
        }
        
        var contains_words = false;
        if (decoded_text.items.len > 0) {
            for (decoded_text.items) |char| {
                if ((char >= 'a' and char <= 'z') or (char >= 'A' and char <= 'Z')) {
                    contains_words = true;
                    break;
                }
            }
        }
        
        const test_passed = has_tokens and has_decoded and valid_token_range and contains_words;
        if (test_passed) passed_tests += 1 else failed_tests += 1;
        
        try stdout.print("  Tokens: {d} (valid range: {s})\n", .{tokens.items.len, if (valid_token_range) "YES" else "NO"});
        try stdout.print("  Decoded length: {d} (contains words: {s})\n", .{decoded_text.items.len, if (contains_words) "YES" else "NO"});
        try stdout.print("  Result: {s}{s}{s}\n\n", .{
            if (test_passed) colors.green else colors.red,
            if (test_passed) "PASSED" else "FAILED",
            colors.reset,
        });
    }
    
    try stdout.print("{s}[TEST 3] SSI Index Retrieval with Score Verification{s}\n", .{colors.bold, colors.reset});
    {
        const query = [_]u32{1, 2, 3, 4, 5};
        var results = try ssi.retrieveTopK(@constCast(&query), config.top_k, allocator);
        defer {
            for (results) |*seg| seg.deinit(allocator);
            allocator.free(results);
        }
        
        var has_valid_scores = true;
        var min_score: f64 = 1.0;
        var max_score: f64 = 0.0;
        for (results) |seg| {
            if (seg.score < 0.0 or seg.score > 1.0) {
                has_valid_scores = false;
            }
            if (seg.score < min_score) min_score = seg.score;
            if (seg.score > max_score) max_score = seg.score;
        }
        
        var has_valid_positions = true;
        for (results) |seg| {
            if (seg.position == std.math.maxInt(u64)) {
                has_valid_positions = false;
                break;
            }
        }
        
        const test_passed = results.len > 0 and has_valid_scores and has_valid_positions;
        if (test_passed) passed_tests += 1 else failed_tests += 1;
        
        try stdout.print("  Retrieved segments: {d}\n", .{results.len});
        if (results.len > 0) {
            try stdout.print("  Score range: [{d:.4}, {d:.4}]\n", .{min_score, max_score});
            try stdout.print("  Valid positions: {s}\n", .{if (has_valid_positions) "YES" else "NO"});
        }
        try stdout.print("  Result: {s}{s}{s}\n\n", .{
            if (test_passed) colors.green else colors.red,
            if (test_passed) "PASSED" else "FAILED",
            colors.reset,
        });
    }
    
    try stdout.print("{s}[TEST 4] Ranker Scoring with Ordering Verification{s}\n", .{colors.bold, colors.reset});
    {
        const query = [_]u32{1, 2, 3, 4, 5};
        var segments = try ssi.retrieveTopK(@constCast(&query), config.top_k, allocator);
        defer {
            for (segments) |*seg| seg.deinit(allocator);
            allocator.free(segments);
        }
        
        var test_passed = false;
        
        if (segments.len > 0) {
            try ranker.rankCandidates(segments, ssi, allocator);
            
            var all_have_scores = true;
            var scores_in_order = true;
            var has_nonzero_score = false;
            
            for (segments) |seg| {
                if (seg.score == 0.0 and segments.len > 1) {
                    all_have_scores = false;
                }
                if (seg.score > 0.0) has_nonzero_score = true;
            }
            
            var i: usize = 0;
            while (i + 1 < segments.len) : (i += 1) {
                if (segments[i].score < segments[i + 1].score) {
                    scores_in_order = false;
                    break;
                }
            }
            
            test_passed = all_have_scores and scores_in_order and has_nonzero_score;
            
            try stdout.print("  Ranked segments: {d}\n", .{segments.len});
            try stdout.print("  All have scores: {s}\n", .{if (all_have_scores) "YES" else "NO"});
            try stdout.print("  Scores in descending order: {s}\n", .{if (scores_in_order) "YES" else "NO"});
            try stdout.print("  Has nonzero scores: {s}\n", .{if (has_nonzero_score) "YES" else "NO"});
            
            if (segments.len > 0 and segments.len <= 5) {
                try stdout.writeAll("  Scores: [");
                i = 0;
                while (i < segments.len) : (i += 1) {
                    try stdout.print("{d:.4}", .{segments[i].score});
                    if (i + 1 < segments.len) try stdout.writeAll(", ");
                }
                try stdout.writeAll("]\n");
            }
        } else {
            try stdout.writeAll("  No segments to rank\n");
        }
        
        if (test_passed) passed_tests += 1 else failed_tests += 1;
        
        try stdout.print("  Result: {s}{s}{s}\n\n", .{
            if (test_passed) colors.green else colors.red,
            if (test_passed) "PASSED" else "FAILED",
            colors.reset,
        });
    }
    
    try stdout.print("{s}[TEST 5] Parameter Extraction and Update with Modification{s}\n", .{colors.bold, colors.reset});
    {
        const total_params = calculateTotalParams(config.embedding_dim, config.rsf_layers);
        var params_original = try extractParameters(allocator, rsf, total_params);
        defer params_original.deinit();
        
        var params_modified = try params_original.copy(allocator);
        defer params_modified.deinit();
        
        var i: usize = 0;
        while (i < params_modified.data.len) : (i += 1) {
            params_modified.data[i] += 0.01;
        }
        
        try updateRSFParameters(rsf, &params_modified);
        
        var params_check = try extractParameters(allocator, rsf, total_params);
        defer params_check.deinit();
        
        var params_updated = true;
        var update_diff_sum: f64 = 0.0;
        i = 0;
        while (i < params_modified.data.len) : (i += 1) {
            const diff = math.fabs(params_modified.data[i] - params_check.data[i]);
            update_diff_sum += diff;
            if (diff > 1e-5) {
                params_updated = false;
            }
        }
        const avg_diff = update_diff_sum / @as(f64, @floatFromInt(params_modified.data.len));
        
        var params_actually_changed = true;
        i = 0;
        while (i < params_original.data.len) : (i += 1) {
            if (math.fabs(params_original.data[i] - params_check.data[i]) < 1e-6) {
                params_actually_changed = false;
                break;
            }
        }
        
        const test_passed = params_updated and params_actually_changed;
        if (test_passed) passed_tests += 1 else failed_tests += 1;
        
        try stdout.print("  Parameters updated correctly: {s}\n", .{if (params_updated) "YES" else "NO"});
        try stdout.print("  Parameters actually changed: {s}\n", .{if (params_actually_changed) "YES" else "NO"});
        try stdout.print("  Average update difference: {d:.8}\n", .{avg_diff});
        try stdout.print("  Result: {s}{s}{s}\n\n", .{
            if (test_passed) colors.green else colors.red,
            if (test_passed) "PASSED" else "FAILED",
            colors.reset,
        });
    }
    
    try stdout.print("{s}[TEST 6] Gradient Clipping with Precise Verification{s}\n", .{colors.bold, colors.reset});
    {
        const timestamp = time.milliTimestamp();
        const seed = if (timestamp >= 0) @as(u64, @intCast(timestamp)) else @as(u64, @intCast(-timestamp));
        const test_seed = seed +% 54321;
        
        var test_grads = try Tensor.randomNormal(allocator, &.{100}, 0.0, 10.0, test_seed);
        defer test_grads.deinit();
        
        var norm_before_sq: f64 = 0.0;
        for (test_grads.data) |val| {
            const v = @as(f64, val);
            norm_before_sq += v * v;
        }
        const norm_before = @sqrt(norm_before_sq);
        
        const clip_norm: f32 = 5.0;
        clipGradients(&test_grads, clip_norm);
        
        var norm_after_sq: f64 = 0.0;
        for (test_grads.data) |val| {
            const v = @as(f64, val);
            norm_after_sq += v * v;
        }
        const norm_after = @sqrt(norm_after_sq);
        
        const clipped_correctly = norm_after <= @as(f64, clip_norm) * 1.001;
        const was_clipped = norm_before > @as(f64, clip_norm);
        const tolerance_ok = norm_after <= @as(f64, clip_norm) + 0.01;
        
        const test_passed = clipped_correctly and tolerance_ok;
        if (test_passed) passed_tests += 1 else failed_tests += 1;
        
        try stdout.print("  Norm before clipping: {d:.6}\n", .{norm_before});
        try stdout.print("  Norm after clipping: {d:.6}\n", .{norm_after});
        try stdout.print("  Target norm: {d:.6}\n", .{clip_norm});
        try stdout.print("  Was clipped: {s}\n", .{if (was_clipped) "YES" else "NO"});
        try stdout.print("  Within tolerance: {s}\n", .{if (tolerance_ok) "YES" else "NO"});
        try stdout.print("  Result: {s}{s}{s}\n\n", .{
            if (test_passed) colors.green else colors.red,
            if (test_passed) "PASSED" else "FAILED",
            colors.reset,
        });
    }
    
    try stdout.writeAll("\n========================================\n");
    try stdout.print("INTEGRATION TESTS COMPLETED: {s}{d}/{d} PASSED{s}\n", .{
        if (passed_tests == total_tests) colors.green else colors.yellow,
        passed_tests,
        total_tests,
        colors.reset,
    });
    try stdout.writeAll("========================================\n");
}
