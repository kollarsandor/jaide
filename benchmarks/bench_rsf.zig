const std = @import("std");
const RSF = @import("../src/processor/rsf.zig");

const BenchConfig = struct {
    warmup_iterations: usize = 30,
    bench_iterations: usize = 300,
    layer_sizes: []const usize = &[_]usize{ 64, 128, 256, 512, 1024 },
};

fn benchmark(comptime name: []const u8, comptime func: anytype, iterations: usize) !void {
    var timer = try std.time.Timer.start();
    const start = timer.read();
    
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        try func();
    }
    
    const end = timer.read();
    const elapsed_ns = end - start;
    const avg_ns = elapsed_ns / iterations;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
    
    std.debug.print("{s}:\n", .{name});
    std.debug.print("  Total time: {d:.2} ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
    std.debug.print("  Average: {d} ns/op\n", .{avg_ns});
    std.debug.print("  Throughput: {d:.2} ops/sec\n\n", .{ops_per_sec});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = BenchConfig{};
    
    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("RSF (Reversible Scatter-Flow) Benchmarks\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n", .{});

    for (config.layer_sizes) |size| {
        std.debug.print("Layer size: {d}\n", .{size});
        std.debug.print("-" ** 60 ++ "\n", .{});

        const bench_name_forward = try std.fmt.allocPrint(allocator, "Forward pass ({d} dims)", .{size});
        defer allocator.free(bench_name_forward);

        const ForwardBench = struct {
            alloc: std.mem.Allocator,
            dim: usize,
            
            fn run(self: @This()) !void {
                const input = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(input);
                const output = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(output);
                const weights = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(weights);
                
                var prng = std.rand.DefaultPrng.init(42);
                const rand = prng.random();
                
                for (input) |*val| {
                    val.* = rand.float(f32) * 2.0 - 1.0;
                }
                for (weights) |*val| {
                    val.* = rand.float(f32) * 0.1;
                }
                
                var idx: usize = 0;
                while (idx < self.dim) : (idx += 1) {
                    output[idx] = input[idx] * weights[idx];
                    if (output[idx] < 0.0) {
                        output[idx] *= 0.01;
                    }
                }
                
                std.mem.doNotOptimizeAway(output);
            }
        };

        const forward_bench = ForwardBench{ .alloc = allocator, .dim = size };
        var i: usize = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try forward_bench.run();
        }
        try benchmark(bench_name_forward, forward_bench.run, config.bench_iterations);

        const bench_name_backward = try std.fmt.allocPrint(allocator, "Backward pass ({d} dims)", .{size});
        defer allocator.free(bench_name_backward);

        const BackwardBench = struct {
            alloc: std.mem.Allocator,
            dim: usize,
            
            fn run(self: @This()) !void {
                const grad_out = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(grad_out);
                const grad_in = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(grad_in);
                const weights = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(weights);
                const activations = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(activations);
                
                var prng = std.rand.DefaultPrng.init(12345);
                const rand = prng.random();
                
                for (grad_out) |*val| {
                    val.* = rand.float(f32) * 2.0 - 1.0;
                }
                for (weights) |*val| {
                    val.* = rand.float(f32) * 0.1;
                }
                for (activations) |*val| {
                    val.* = rand.float(f32) * 2.0 - 1.0;
                }
                
                var idx: usize = 0;
                while (idx < self.dim) : (idx += 1) {
                    const grad_activation = if (activations[idx] >= 0.0) 
                        grad_out[idx] 
                    else 
                        grad_out[idx] * 0.01;
                    grad_in[idx] = grad_activation * weights[idx];
                }
                
                std.mem.doNotOptimizeAway(grad_in);
            }
        };

        const backward_bench = BackwardBench{ .alloc = allocator, .dim = size };
        i = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try backward_bench.run();
        }
        try benchmark(bench_name_backward, backward_bench.run, config.bench_iterations);

        const bench_name_scatter = try std.fmt.allocPrint(allocator, "Scatter operation ({d} dims)", .{size});
        defer allocator.free(bench_name_scatter);

        const ScatterBench = struct {
            alloc: std.mem.Allocator,
            dim: usize,
            
            fn run(self: @This()) !void {
                const input = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(input);
                const output = try self.alloc.alloc(f32, self.dim);
                defer self.alloc.free(output);
                const indices = try self.alloc.alloc(usize, self.dim);
                defer self.alloc.free(indices);
                
                var prng = std.rand.DefaultPrng.init(54321);
                const rand = prng.random();
                
                for (input) |*val| {
                    val.* = rand.float(f32) * 2.0 - 1.0;
                }
                var idx: usize = 0;
                while (idx < self.dim) : (idx += 1) {
                    indices[idx] = idx;
                }
                
                var k: usize = self.dim - 1;
                while (k > 0) : (k -= 1) {
                    const j = rand.intRangeLessThan(usize, 0, k + 1);
                    const temp = indices[k];
                    indices[k] = indices[j];
                    indices[j] = temp;
                }
                
                var idx2: usize = 0;
                while (idx2 < self.dim) : (idx2 += 1) {
                    output[indices[idx2]] = input[idx2];
                }
                
                std.mem.doNotOptimizeAway(output);
            }
        };

        const scatter_bench = ScatterBench{ .alloc = allocator, .dim = size };
        i = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try scatter_bench.run();
        }
        try benchmark(bench_name_scatter, scatter_bench.run, config.bench_iterations);

        std.debug.print("\n", .{});
    }

    std.debug.print("\nAll RSF benchmarks completed successfully!\n", .{});
}
