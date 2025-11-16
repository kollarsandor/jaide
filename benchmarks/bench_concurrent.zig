const std = @import("std");
const Thread = std.Thread;
const Allocator = std.mem.Allocator;
const deps = @import("ssi");
const SSI = deps.SSI;
const RSF = deps.RSF;
const Tensor = deps.Tensor;
const RankedSegment = deps.RankedSegment;

const NUM_THREADS = 8;
const SSI_OPS_PER_THREAD = 500;
const RSF_OPS_PER_THREAD = 100;
const RANKING_OPS_PER_THREAD = 200;

const BenchConfig = struct {
    warmup_iterations: usize = 5,
    bench_iterations: usize = 10,
};

const SSIInsertContext = struct {
    ssi: *SSI,
    thread_id: usize,
    allocator: Allocator,
    mutex: *std.Thread.Mutex,
    barrier: *std.atomic.Atomic(usize),
    total_threads: usize,
    ops_completed: *std.atomic.Atomic(usize),
};

fn ssiInsertWorker(ctx: SSIInsertContext) !void {
    var prng = std.rand.DefaultPrng.init(@intCast(ctx.thread_id * 54321));
    const rand = prng.random();
    
    _ = ctx.barrier.fetchAdd(1, .SeqCst);
    while (ctx.barrier.load(.SeqCst) < ctx.total_threads) {
        std.Thread.yield() catch {};
    }
    
    var ops: usize = 0;
    while (ops < SSI_OPS_PER_THREAD) : (ops += 1) {
        const token_count = rand.intRangeAtMost(usize, 5, 20);
        var tokens = try ctx.allocator.alloc(u32, token_count);
        defer ctx.allocator.free(tokens);
        
        var i: usize = 0;
        while (i < token_count) : (i += 1) {
            tokens[i] = rand.int(u32) % 50000;
        }
        
        const position = @as(u64, ctx.thread_id) * 1000000 + ops;
        const is_anchor = rand.boolean();
        
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        
        try ctx.ssi.addSequence(tokens, position, is_anchor);
        
        _ = ctx.ops_completed.fetchAdd(1, .SeqCst);
    }
}

const RSFForwardContext = struct {
    rsf: *RSF,
    input_tensors: []Tensor,
    thread_id: usize,
    allocator: Allocator,
    barrier: *std.atomic.Atomic(usize),
    total_threads: usize,
    ops_completed: *std.atomic.Atomic(usize),
};

fn rsfForwardWorker(ctx: RSFForwardContext) !void {
    _ = ctx.barrier.fetchAdd(1, .SeqCst);
    while (ctx.barrier.load(.SeqCst) < ctx.total_threads) {
        std.Thread.yield() catch {};
    }
    
    var ops: usize = 0;
    while (ops < RSF_OPS_PER_THREAD) : (ops += 1) {
        const tensor_idx = ctx.thread_id;
        var tensor_copy = try ctx.input_tensors[tensor_idx].copy(ctx.allocator);
        defer tensor_copy.deinit();
        
        try ctx.rsf.forward(&tensor_copy);
        
        _ = ctx.ops_completed.fetchAdd(1, .SeqCst);
    }
}

const RankingContext = struct {
    segments: []RankedSegment,
    thread_id: usize,
    barrier: *std.atomic.Atomic(usize),
    total_threads: usize,
    ops_completed: *std.atomic.Atomic(usize),
    chunk_size: usize,
};

fn rankingWorker(ctx: RankingContext) void {
    _ = ctx.barrier.fetchAdd(1, .SeqCst);
    while (ctx.barrier.load(.SeqCst) < ctx.total_threads) {
        std.Thread.yield() catch {};
    }
    
    const start_idx = ctx.thread_id * ctx.chunk_size;
    const end_idx = @min(start_idx + ctx.chunk_size, ctx.segments.len);
    
    var ops: usize = 0;
    while (ops < RANKING_OPS_PER_THREAD) : (ops += 1) {
        var i = start_idx;
        while (i < end_idx) : (i += 1) {
            var j = i + 1;
            while (j < end_idx) : (j += 1) {
                const cmp = ctx.segments[i].compare(ctx.segments[j]);
                if (cmp > 0) {
                    const temp = ctx.segments[i];
                    ctx.segments[i] = ctx.segments[j];
                    ctx.segments[j] = temp;
                }
            }
        }
        
        _ = ctx.ops_completed.fetchAdd(1, .SeqCst);
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        .thread_safe = true,
    }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = BenchConfig{};
    
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("CONCURRENT BENCHMARKS - JAIDE v40\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("Threads: {d}\n", .{NUM_THREADS});
    std.debug.print("SSI operations per thread: {d}\n", .{SSI_OPS_PER_THREAD});
    std.debug.print("RSF operations per thread: {d}\n", .{RSF_OPS_PER_THREAD});
    std.debug.print("Ranking operations per thread: {d}\n", .{RANKING_OPS_PER_THREAD});
    std.debug.print("-" ** 80 ++ "\n\n", .{});

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("BENCHMARK 1: Concurrent SSI Insertions\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    const SSIBench = struct {
        alloc: Allocator,
        num_threads: usize,
        
        fn run(self: @This()) !void {
            var ssi = SSI.init(self.alloc);
            defer ssi.deinit();
            
            var mutex = std.Thread.Mutex{};
            var barrier = std.atomic.Atomic(usize).init(0);
            var ops_completed = std.atomic.Atomic(usize).init(0);
            
            var threads = try self.alloc.alloc(Thread, self.num_threads);
            defer self.alloc.free(threads);
            
            var t: usize = 0;
            while (t < self.num_threads) : (t += 1) {
                const ctx = SSIInsertContext{
                    .ssi = &ssi,
                    .thread_id = t,
                    .allocator = self.alloc,
                    .mutex = &mutex,
                    .barrier = &barrier,
                    .total_threads = self.num_threads,
                    .ops_completed = &ops_completed,
                };
                threads[t] = try Thread.spawn(.{}, ssiInsertWorker, .{ctx});
            }
            
            t = 0;
            while (t < self.num_threads) : (t += 1) {
                threads[t].join();
            }
        }
    };

    const ssi_bench = SSIBench{ .alloc = allocator, .num_threads = NUM_THREADS };
    
    var i: usize = 0;
    while (i < config.warmup_iterations) : (i += 1) {
        try ssi_bench.run();
    }
    
    const start_time = std.time.nanoTimestamp();
    i = 0;
    while (i < config.bench_iterations) : (i += 1) {
        try ssi_bench.run();
    }
    const end_time = std.time.nanoTimestamp();
    const elapsed_ns = @as(u64, @intCast(end_time - start_time));
    const avg_ns = elapsed_ns / config.bench_iterations;
    const ops_per_sec = @as(f64, @floatFromInt(config.bench_iterations)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
    
    std.debug.print("Concurrent SSI Insertions:\n", .{});
    std.debug.print("  Total time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
    std.debug.print("  Average: {d} ns/op\n", .{avg_ns});
    std.debug.print("  Throughput: {d:.2} ops/sec\n\n", .{ops_per_sec});

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("BENCHMARK 2: Parallel RSF Forward Passes\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    const RSFBench = struct {
        alloc: Allocator,
        num_threads: usize,
        
        fn run(self: @This()) !void {
            const dim = 32;
            const num_layers = 4;
            var rsf = try RSF.init(self.alloc, dim, num_layers);
            defer rsf.deinit();
            
            var input_tensors = try self.alloc.alloc(Tensor, self.num_threads);
            defer {
                for (input_tensors) |*t| {
                    t.deinit();
                }
                self.alloc.free(input_tensors);
            }
            
            var t: usize = 0;
            while (t < self.num_threads) : (t += 1) {
                input_tensors[t] = try Tensor.randomNormal(self.alloc, &.{ 16, dim * 2 }, 0, 1, @intCast(t + 1000));
            }
            
            var barrier = std.atomic.Atomic(usize).init(0);
            var ops_completed = std.atomic.Atomic(usize).init(0);
            
            var threads = try self.alloc.alloc(Thread, self.num_threads);
            defer self.alloc.free(threads);
            
            t = 0;
            while (t < self.num_threads) : (t += 1) {
                const ctx = RSFForwardContext{
                    .rsf = &rsf,
                    .input_tensors = input_tensors,
                    .thread_id = t,
                    .allocator = self.alloc,
                    .barrier = &barrier,
                    .total_threads = self.num_threads,
                    .ops_completed = &ops_completed,
                };
                threads[t] = try Thread.spawn(.{}, rsfForwardWorker, .{ctx});
            }
            
            t = 0;
            while (t < self.num_threads) : (t += 1) {
                threads[t].join();
            }
        }
    };

    const rsf_bench = RSFBench{ .alloc = allocator, .num_threads = NUM_THREADS };
    
    i = 0;
    while (i < config.warmup_iterations) : (i += 1) {
        try rsf_bench.run();
    }
    
    const start_time2 = std.time.nanoTimestamp();
    i = 0;
    while (i < config.bench_iterations) : (i += 1) {
        try rsf_bench.run();
    }
    const end_time2 = std.time.nanoTimestamp();
    const elapsed_ns2 = @as(u64, @intCast(end_time2 - start_time2));
    const avg_ns2 = elapsed_ns2 / config.bench_iterations;
    const ops_per_sec2 = @as(f64, @floatFromInt(config.bench_iterations)) / (@as(f64, @floatFromInt(elapsed_ns2)) / 1_000_000_000.0);
    
    std.debug.print("Parallel RSF Forward Passes:\n", .{});
    std.debug.print("  Total time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns2)) / 1_000_000.0});
    std.debug.print("  Average: {d} ns/op\n", .{avg_ns2});
    std.debug.print("  Throughput: {d:.2} ops/sec\n\n", .{ops_per_sec2});

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("BENCHMARK 3: Multi-threaded Ranking\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    const RankingBench = struct {
        alloc: Allocator,
        num_threads: usize,
        
        fn run(self: @This()) !void {
            const num_segments = 1000;
            var segments = try self.alloc.alloc(RankedSegment, num_segments);
            defer {
                for (segments) |*seg| {
                    seg.deinit(self.alloc);
                }
                self.alloc.free(segments);
            }
            
            var prng = std.rand.DefaultPrng.init(9999);
            const rand = prng.random();
            
            var s: usize = 0;
            while (s < num_segments) : (s += 1) {
                const token_count = rand.intRangeAtMost(usize, 3, 10);
                var tokens = try self.alloc.alloc(u32, token_count);
                var j: usize = 0;
                while (j < token_count) : (j += 1) {
                    tokens[j] = rand.int(u32) % 10000;
                }
                
                segments[s] = RankedSegment{
                    .tokens = tokens,
                    .score = rand.float(f32),
                    .position = @intCast(s),
                    .anchor = rand.boolean(),
                };
            }
            
            var barrier = std.atomic.Atomic(usize).init(0);
            var ops_completed = std.atomic.Atomic(usize).init(0);
            
            var threads = try self.alloc.alloc(Thread, self.num_threads);
            defer self.alloc.free(threads);
            
            const chunk_size = (num_segments + self.num_threads - 1) / self.num_threads;
            
            var t: usize = 0;
            while (t < self.num_threads) : (t += 1) {
                const ctx = RankingContext{
                    .segments = segments,
                    .thread_id = t,
                    .barrier = &barrier,
                    .total_threads = self.num_threads,
                    .ops_completed = &ops_completed,
                    .chunk_size = chunk_size,
                };
                threads[t] = try Thread.spawn(.{}, rankingWorker, .{ctx});
            }
            
            t = 0;
            while (t < self.num_threads) : (t += 1) {
                threads[t].join();
            }
        }
    };

    const ranking_bench = RankingBench{ .alloc = allocator, .num_threads = NUM_THREADS };
    
    i = 0;
    while (i < config.warmup_iterations) : (i += 1) {
        try ranking_bench.run();
    }
    
    const start_time3 = std.time.nanoTimestamp();
    i = 0;
    while (i < config.bench_iterations) : (i += 1) {
        try ranking_bench.run();
    }
    const end_time3 = std.time.nanoTimestamp();
    const elapsed_ns3 = @as(u64, @intCast(end_time3 - start_time3));
    const avg_ns3 = elapsed_ns3 / config.bench_iterations;
    const ops_per_sec3 = @as(f64, @floatFromInt(config.bench_iterations)) / (@as(f64, @floatFromInt(elapsed_ns3)) / 1_000_000_000.0);
    
    std.debug.print("Multi-threaded Ranking:\n", .{});
    std.debug.print("  Total time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns3)) / 1_000_000.0});
    std.debug.print("  Average: {d} ns/op\n", .{avg_ns3});
    std.debug.print("  Throughput: {d:.2} ops/sec\n\n", .{ops_per_sec3});

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("BENCHMARK 4: Mixed Concurrent Workload\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});

    const MixedBench = struct {
        alloc: Allocator,
        
        fn run(self: @This()) !void {
            var ssi = SSI.init(self.alloc);
            defer ssi.deinit();
            
            var rsf = try RSF.init(self.alloc, 16, 2);
            defer rsf.deinit();
            
            const num_ops = 100;
            var prng = std.rand.DefaultPrng.init(7777);
            const rand = prng.random();
            
            var op: usize = 0;
            while (op < num_ops) : (op += 1) {
                const op_type = rand.intRangeAtMost(u8, 0, 2);
                
                switch (op_type) {
                    0 => {
                        const token_count = rand.intRangeAtMost(usize, 5, 15);
                        var tokens = try self.alloc.alloc(u32, token_count);
                        defer self.alloc.free(tokens);
                        
                        var j: usize = 0;
                        while (j < token_count) : (j += 1) {
                            tokens[j] = rand.int(u32) % 30000;
                        }
                        
                        try ssi.addSequence(tokens, @intCast(op), rand.boolean());
                    },
                    1 => {
                        var tensor = try Tensor.randomNormal(self.alloc, &.{ 4, 32 }, 0, 1, @intCast(op + 5000));
                        defer tensor.deinit();
                        
                        try rsf.forward(&tensor);
                    },
                    else => {
                        const query_size = rand.intRangeAtMost(usize, 3, 8);
                        var query = try self.alloc.alloc(u32, query_size);
                        defer self.alloc.free(query);
                        
                        var j: usize = 0;
                        while (j < query_size) : (j += 1) {
                            query[j] = rand.int(u32) % 30000;
                        }
                        
                        const results = try ssi.retrieveTopK(query, 10, self.alloc);
                        for (results) |*result| {
                            result.deinit(self.alloc);
                        }
                        self.alloc.free(results);
                    },
                }
            }
        }
    };

    const mixed_bench = MixedBench{ .alloc = allocator };
    
    i = 0;
    while (i < config.warmup_iterations) : (i += 1) {
        try mixed_bench.run();
    }
    
    const start_time4 = std.time.nanoTimestamp();
    i = 0;
    while (i < config.bench_iterations) : (i += 1) {
        try mixed_bench.run();
    }
    const end_time4 = std.time.nanoTimestamp();
    const elapsed_ns4 = @as(u64, @intCast(end_time4 - start_time4));
    const avg_ns4 = elapsed_ns4 / config.bench_iterations;
    const ops_per_sec4 = @as(f64, @floatFromInt(config.bench_iterations)) / (@as(f64, @floatFromInt(elapsed_ns4)) / 1_000_000_000.0);
    
    std.debug.print("Mixed Concurrent Workload:\n", .{});
    std.debug.print("  Total time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns4)) / 1_000_000.0});
    std.debug.print("  Average: {d} ns/op\n", .{avg_ns4});
    std.debug.print("  Throughput: {d:.2} ops/sec\n\n", .{ops_per_sec4});

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("All concurrent benchmarks completed successfully!\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
}

test "concurrent SSI operations" {
    const allocator = std.testing.allocator;
    
    var ssi = SSI.init(allocator);
    defer ssi.deinit();
    
    var mutex = std.Thread.Mutex{};
    var barrier = std.atomic.Atomic(usize).init(0);
    var ops_completed = std.atomic.Atomic(usize).init(0);
    
    const num_threads = 4;
    
    var threads = try allocator.alloc(Thread, num_threads);
    defer allocator.free(threads);
    
    var t: usize = 0;
    while (t < num_threads) : (t += 1) {
        const ctx = SSIInsertContext{
            .ssi = &ssi,
            .thread_id = t,
            .allocator = allocator,
            .mutex = &mutex,
            .barrier = &barrier,
            .total_threads = num_threads,
            .ops_completed = &ops_completed,
        };
        threads[t] = try Thread.spawn(.{}, ssiInsertWorker, .{ctx});
    }
    
    t = 0;
    while (t < num_threads) : (t += 1) {
        threads[t].join();
    }
    
    const total_ops = ops_completed.load(.SeqCst);
    try std.testing.expect(total_ops > 0);
}

test "parallel RSF forward" {
    const allocator = std.testing.allocator;
    
    const dim = 16;
    const num_layers = 2;
    var rsf = try RSF.init(allocator, dim, num_layers);
    defer rsf.deinit();
    
    const num_threads = 4;
    var input_tensors = try allocator.alloc(Tensor, num_threads);
    defer {
        for (input_tensors) |*tensor| {
            tensor.deinit();
        }
        allocator.free(input_tensors);
    }
    
    var t: usize = 0;
    while (t < num_threads) : (t += 1) {
        input_tensors[t] = try Tensor.randomNormal(allocator, &.{ 8, dim * 2 }, 0, 1, @intCast(t + 2000));
    }
    
    var barrier = std.atomic.Atomic(usize).init(0);
    var ops_completed = std.atomic.Atomic(usize).init(0);
    
    var threads = try allocator.alloc(Thread, num_threads);
    defer allocator.free(threads);
    
    t = 0;
    while (t < num_threads) : (t += 1) {
        const ctx = RSFForwardContext{
            .rsf = &rsf,
            .input_tensors = input_tensors,
            .thread_id = t,
            .allocator = allocator,
            .barrier = &barrier,
            .total_threads = num_threads,
            .ops_completed = &ops_completed,
        };
        threads[t] = try Thread.spawn(.{}, rsfForwardWorker, .{ctx});
    }
    
    t = 0;
    while (t < num_threads) : (t += 1) {
        threads[t].join();
    }
    
    const total_ops = ops_completed.load(.SeqCst);
    try std.testing.expect(total_ops > 0);
}
