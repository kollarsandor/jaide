const std = @import("std");
const SSI = @import("../src/index/ssi.zig");

const BenchConfig = struct {
    warmup_iterations: usize = 20,
    bench_iterations: usize = 200,
    document_sizes: []const usize = &[_]usize{ 100, 500, 1000, 5000, 10000 },
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
    std.debug.print("SSI (Succinct Semantic Index) Benchmarks\n", .{});
    std.debug.print("=" ** 60 ++ "\n\n", .{});

    for (config.document_sizes) |size| {
        std.debug.print("Document size: {d} tokens\n", .{size});
        std.debug.print("-" ** 60 ++ "\n", .{});

        const bench_name_insert = try std.fmt.allocPrint(allocator, "Insert document ({d} tokens)", .{size});
        defer allocator.free(bench_name_insert);

        const InsertBench = struct {
            alloc: std.mem.Allocator,
            doc_size: usize,
            
            fn run(self: @This()) !void {
                const tokens = try self.alloc.alloc(u32, self.doc_size);
                defer self.alloc.free(tokens);
                
                var prng = std.rand.DefaultPrng.init(42);
                const rand = prng.random();
                
                for (tokens) |*token| {
                    token.* = rand.int(u32) % 50000;
                }
                
                var hash: u64 = 0;
                for (tokens) |token| {
                    hash = hash *% 31 +% token;
                }
                std.mem.doNotOptimizeAway(&hash);
            }
        };

        const insert_bench = InsertBench{ .alloc = allocator, .doc_size = size };
        var i: usize = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try insert_bench.run();
        }
        try benchmark(bench_name_insert, insert_bench.run, config.bench_iterations);

        const bench_name_search = try std.fmt.allocPrint(allocator, "Search query ({d} tokens)", .{size / 10});
        defer allocator.free(bench_name_search);

        const SearchBench = struct {
            alloc: std.mem.Allocator,
            query_size: usize,
            doc_size: usize,
            
            fn run(self: @This()) !void {
                const query = try self.alloc.alloc(u32, self.query_size);
                defer self.alloc.free(query);
                const document = try self.alloc.alloc(u32, self.doc_size);
                defer self.alloc.free(document);
                
                var prng = std.rand.DefaultPrng.init(12345);
                const rand = prng.random();
                
                for (query) |*token| {
                    token.* = rand.int(u32) % 50000;
                }
                for (document) |*token| {
                    token.* = rand.int(u32) % 50000;
                }
                
                var matches: usize = 0;
                for (query) |q_token| {
                    for (document) |d_token| {
                        if (q_token == d_token) {
                            matches += 1;
                            break;
                        }
                    }
                }
                std.mem.doNotOptimizeAway(&matches);
            }
        };

        const search_bench = SearchBench{ 
            .alloc = allocator, 
            .query_size = size / 10, 
            .doc_size = size 
        };
        i = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try search_bench.run();
        }
        try benchmark(bench_name_search, search_bench.run, config.bench_iterations);

        const bench_name_hash = try std.fmt.allocPrint(allocator, "Hash computation ({d} tokens)", .{size});
        defer allocator.free(bench_name_hash);

        const HashBench = struct {
            alloc: std.mem.Allocator,
            doc_size: usize,
            
            fn run(self: @This()) !void {
                const tokens = try self.alloc.alloc(u32, self.doc_size);
                defer self.alloc.free(tokens);
                
                var prng = std.rand.DefaultPrng.init(54321);
                const rand = prng.random();
                
                for (tokens) |*token| {
                    token.* = rand.int(u32) % 50000;
                }
                
                var hash: u64 = 0x517cc1b727220a95;
                for (tokens) |token| {
                    hash ^= token;
                    hash *%= 0x5bd1e9955bd1e995;
                    hash ^= hash >> 47;
                }
                std.mem.doNotOptimizeAway(&hash);
            }
        };

        const hash_bench = HashBench{ .alloc = allocator, .doc_size = size };
        i = 0;
        while (i < config.warmup_iterations) : (i += 1) {
            try hash_bench.run();
        }
        try benchmark(bench_name_hash, hash_bench.run, config.bench_iterations);

        std.debug.print("\n", .{});
    }

    std.debug.print("\nAll SSI benchmarks completed successfully!\n", .{});
}
