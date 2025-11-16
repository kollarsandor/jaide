const std = @import("std");
const Memory = @import("../src/core/memory.zig");

const BenchConfig = struct {
    warmup_iterations: usize = 100,
    bench_iterations: usize = 1000,
    allocation_sizes: []const usize = &[_]usize{ 16, 64, 256, 1024, 4096, 16384, 65536 },
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
    std.debug.print("Memory Allocation Benchmarks\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n", .{});

    for (config.allocation_sizes) |size| {
        const bench_name = try std.fmt.allocPrint(allocator, "Allocate {d} bytes", .{size});
        defer allocator.free(bench_name);

        const AllocBench = struct {
            alloc: std.mem.Allocator,
            sz: usize,
            
            fn run(self: @This()) !void {
                const mem = try self.alloc.alloc(u8, self.sz);
                defer self.alloc.free(mem);
                @memset(mem, 0);
            }
        };

        const bench_instance = AllocBench{ .alloc = allocator, .sz = size };
        
        var i: usize = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try bench_instance.run();
        }

        try benchmark(bench_name, bench_instance.run, config.bench_iterations);
    }

    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("Aligned Allocation Benchmarks\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n");

    const alignments = [_]usize{ 8, 16, 32, 64 };
    for (alignments) |alignment| {
        const bench_name = try std.fmt.allocPrint(allocator, "Aligned alloc (4096 bytes, {d}-byte alignment)", .{alignment});
        defer allocator.free(bench_name);

        const AlignedAllocBench = struct {
            alloc: std.mem.Allocator,
            align_val: usize,
            
            fn run(self: @This()) !void {
                const mem = try self.alloc.alignedAlloc(u8, self.align_val, 4096);
                defer self.alloc.free(mem);
                @memset(mem, 0);
            }
        };

        const bench_instance = AlignedAllocBench{ .alloc = allocator, .align_val = alignment };
        
        var i: usize = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try bench_instance.run();
        }

        try benchmark(bench_name, bench_instance.run, config.bench_iterations);
    }

    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("Reallocation Benchmarks\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n");

    const realloc_sizes = [_][2]usize{
        .{ 64, 128 },
        .{ 256, 512 },
        .{ 1024, 2048 },
        .{ 4096, 8192 },
    };

    for (realloc_sizes) |size_pair| {
        const bench_name = try std.fmt.allocPrint(allocator, "Realloc {d} -> {d} bytes", .{ size_pair[0], size_pair[1] });
        defer allocator.free(bench_name);

        const ReallocBench = struct {
            alloc: std.mem.Allocator,
            old_size: usize,
            new_size: usize,
            
            fn run(self: @This()) !void {
                var mem = try self.alloc.alloc(u8, self.old_size);
                @memset(mem, 0);
                mem = try self.alloc.realloc(mem, self.new_size);
                self.alloc.free(mem);
            }
        };

        const bench_instance = ReallocBench{ 
            .alloc = allocator, 
            .old_size = size_pair[0], 
            .new_size = size_pair[1] 
        };
        
        var i: usize = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try bench_instance.run();
        }

        try benchmark(bench_name, bench_instance.run, config.bench_iterations);
    }

    std.debug.print("\nAll benchmarks completed successfully!\n", .{});
}
