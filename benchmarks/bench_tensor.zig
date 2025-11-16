const std = @import("std");
const Tensor = @import("../src/core/tensor.zig");

const BenchConfig = struct {
    warmup_iterations: usize = 50,
    bench_iterations: usize = 500,
    tensor_sizes: []const usize = &[_]usize{ 64, 128, 256, 512, 1024, 2048 },
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
    std.debug.print("Tensor Operation Benchmarks\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n", .{});

    for (config.tensor_sizes) |size| {
        std.debug.print("Tensor size: {d}x{d}\n", .{ size, size });
        std.debug.print("-" ** 60 ++ "\n", .{});

        const bench_name_add = try std.fmt.allocPrint(allocator, "Element-wise addition ({d}x{d})", .{ size, size });
        defer allocator.free(bench_name_add);

        const AddBench = struct {
            alloc: std.mem.Allocator,
            sz: usize,
            
            fn run(self: @This()) !void {
                const total = self.sz * self.sz;
                const a = try self.alloc.alloc(f32, total);
                defer self.alloc.free(a);
                const b = try self.alloc.alloc(f32, total);
                defer self.alloc.free(b);
                const c = try self.alloc.alloc(f32, total);
                defer self.alloc.free(c);
                
                @memset(a, 1.0);
                @memset(b, 2.0);
                
                var idx: usize = 0;
                while (idx < total) : (idx += 1) {
                    c[idx] = a[idx] + b[idx];
                }
            }
        };

        const add_bench = AddBench{ .alloc = allocator, .sz = size };
        var i: usize = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try add_bench.run();
        }
        try benchmark(bench_name_add, add_bench.run, config.bench_iterations);

        const bench_name_mul = try std.fmt.allocPrint(allocator, "Element-wise multiplication ({d}x{d})", .{ size, size });
        defer allocator.free(bench_name_mul);

        const MulBench = struct {
            alloc: std.mem.Allocator,
            sz: usize,
            
            fn run(self: @This()) !void {
                const total = self.sz * self.sz;
                const a = try self.alloc.alloc(f32, total);
                defer self.alloc.free(a);
                const b = try self.alloc.alloc(f32, total);
                defer self.alloc.free(b);
                const c = try self.alloc.alloc(f32, total);
                defer self.alloc.free(c);
                
                @memset(a, 1.5);
                @memset(b, 2.5);
                
                var idx: usize = 0;
                while (idx < total) : (idx += 1) {
                    c[idx] = a[idx] * b[idx];
                }
            }
        };

        const mul_bench = MulBench{ .alloc = allocator, .sz = size };
        i = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try mul_bench.run();
        }
        try benchmark(bench_name_mul, mul_bench.run, config.bench_iterations);

        const bench_name_reduce = try std.fmt.allocPrint(allocator, "Reduction sum ({d}x{d})", .{ size, size });
        defer allocator.free(bench_name_reduce);

        const ReduceBench = struct {
            alloc: std.mem.Allocator,
            sz: usize,
            
            fn run(self: @This()) !void {
                const total = self.sz * self.sz;
                const a = try self.alloc.alloc(f32, total);
                defer self.alloc.free(a);
                
                @memset(a, 1.0);
                
                var sum: f32 = 0.0;
                for (a) |val| {
                    sum += val;
                }
                std.mem.doNotOptimizeAway(&sum);
            }
        };

        const reduce_bench = ReduceBench{ .alloc = allocator, .sz = size };
        i = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try reduce_bench.run();
        }
        try benchmark(bench_name_reduce, reduce_bench.run, config.bench_iterations);

        std.debug.print("\n", .{});
    }

    std.debug.print("\nAll tensor benchmarks completed successfully!\n", .{});
}
