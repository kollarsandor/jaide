const std = @import("std");

comptime {
    const required_zig = "0.11.0";
    const current_zig = @import("builtin").zig_version_string;
    _ = required_zig;
    _ = current_zig;
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const core_lib = b.addStaticLibrary(.{
        .name = "jaide_core",
        .root_source_file = .{ .path = "src/core/types.zig" },
        .target = target,
        .optimize = optimize,
    });
    core_lib.linkLibC();
    core_lib.linkSystemLibrary("m");
    b.installArtifact(core_lib);

    const jaide_training = b.addExecutable(.{
        .name = "jaide_training",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    jaide_training.linkLibrary(core_lib);
    b.installArtifact(jaide_training);

    const training_step = b.step("training", "Build JAIDE training executable");
    training_step.dependOn(&jaide_training.step);

    const build_step = b.step("build", "Build training executable (default)");
    build_step.dependOn(&jaide_training.step);

    const test_step = b.addTest(.{
        .root_source_file = .{ .path = "src/core/types.zig" },
        .target = target,
        .optimize = optimize,
    });
    test_step.linkLibrary(core_lib);
    const run_tests = b.addRunArtifact(test_step);
    const test_runner = b.step("test", "Run unit tests");
    test_runner.dependOn(&run_tests.step);

    const run_training = b.addRunArtifact(jaide_training);
    run_training.step.dependOn(&jaide_training.step);
    const run_step = b.step("run", "Run training executable");
    run_step.dependOn(&run_training.step);

    const verify_cmd = b.addSystemCommand(&[_][]const u8{
        "bash",
        "scripts/verify_all.sh",
    });
    const verify_step = b.step("verify", "Run all formal verification proofs");
    verify_step.dependOn(&verify_cmd.step);

    const fuzz_memory = b.addExecutable(.{
        .name = "fuzz_memory",
        .root_source_file = .{ .path = "fuzz/fuzz_memory.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    fuzz_memory.linkLibrary(core_lib);
    
    const fuzz_tensor = b.addExecutable(.{
        .name = "fuzz_tensor",
        .root_source_file = .{ .path = "fuzz/fuzz_tensor.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    fuzz_tensor.linkLibrary(core_lib);
    
    const fuzz_ssi = b.addExecutable(.{
        .name = "fuzz_ssi",
        .root_source_file = .{ .path = "fuzz/fuzz_ssi.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    fuzz_ssi.linkLibrary(core_lib);

    const run_fuzz_memory = b.addRunArtifact(fuzz_memory);
    const run_fuzz_tensor = b.addRunArtifact(fuzz_tensor);
    const run_fuzz_ssi = b.addRunArtifact(fuzz_ssi);
    
    const fuzz_step = b.step("fuzz", "Run fuzz tests");
    fuzz_step.dependOn(&run_fuzz_memory.step);
    fuzz_step.dependOn(&run_fuzz_tensor.step);
    fuzz_step.dependOn(&run_fuzz_ssi.step);

    const bench_memory = b.addExecutable(.{
        .name = "bench_memory",
        .root_source_file = .{ .path = "benchmarks/bench_memory.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_memory.linkLibrary(core_lib);
    
    const bench_tensor = b.addExecutable(.{
        .name = "bench_tensor",
        .root_source_file = .{ .path = "benchmarks/bench_tensor.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_tensor.linkLibrary(core_lib);
    
    const bench_ssi = b.addExecutable(.{
        .name = "bench_ssi",
        .root_source_file = .{ .path = "benchmarks/bench_ssi.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_ssi.linkLibrary(core_lib);
    
    const bench_rsf = b.addExecutable(.{
        .name = "bench_rsf",
        .root_source_file = .{ .path = "benchmarks/bench_rsf.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_rsf.linkLibrary(core_lib);

    const run_bench_memory = b.addRunArtifact(bench_memory);
    const run_bench_tensor = b.addRunArtifact(bench_tensor);
    const run_bench_ssi = b.addRunArtifact(bench_ssi);
    const run_bench_rsf = b.addRunArtifact(bench_rsf);
    
    const bench_step = b.step("bench", "Run performance benchmarks");
    bench_step.dependOn(&run_bench_memory.step);
    bench_step.dependOn(&run_bench_tensor.step);
    bench_step.dependOn(&run_bench_ssi.step);
    bench_step.dependOn(&run_bench_rsf.step);

    const sanitize_step = b.step("sanitize", "Build and test with sanitizers");
    const sanitize_test = b.addTest(.{
        .root_source_file = .{ .path = "src/core/types.zig" },
        .target = target,
        .optimize = .Debug,
    });
    sanitize_test.linkLibrary(core_lib);
    const run_sanitize_test = b.addRunArtifact(sanitize_test);
    sanitize_step.dependOn(&run_sanitize_test.step);

    const valgrind_step = b.step("valgrind", "Run tests under Valgrind");
    const valgrind_test = b.addTest(.{
        .root_source_file = .{ .path = "src/core/types.zig" },
        .target = target,
        .optimize = .Debug,
    });
    valgrind_test.linkLibrary(core_lib);
    
    const install_valgrind_test = b.addInstallArtifact(valgrind_test, .{});
    const valgrind_cmd = b.addSystemCommand(&[_][]const u8{
        "valgrind",
        "--leak-check=full",
        "--show-leak-kinds=all",
        "--track-origins=yes",
        "--error-exitcode=1",
    });
    valgrind_cmd.addArtifactArg(valgrind_test);
    valgrind_cmd.step.dependOn(&install_valgrind_test.step);
    valgrind_step.dependOn(&valgrind_cmd.step);

    const stress_refcount = b.addExecutable(.{
        .name = "stress_tensor_refcount",
        .root_source_file = .{ .path = "tests/stress_tensor_refcount.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    stress_refcount.linkLibrary(core_lib);
    stress_refcount.addAnonymousModule("types", .{ .source_file = .{ .path = "src/core/types.zig" } });
    
    const run_stress_refcount = b.addRunArtifact(stress_refcount);
    const stress_step = b.step("stress", "Run multithreaded stress tests");
    stress_step.dependOn(&run_stress_refcount.step);

    const bench_concurrent = b.addExecutable(.{
        .name = "bench_concurrent",
        .root_source_file = .{ .path = "benchmarks/bench_concurrent.zig" },
        .target = target,
        .optimize = .ReleaseFast,
    });
    bench_concurrent.linkLibrary(core_lib);
    bench_concurrent.addAnonymousModule("ssi", .{ .source_file = .{ .path = "src/bench_deps.zig" } });
    bench_concurrent.addAnonymousModule("rsf", .{ .source_file = .{ .path = "src/bench_deps.zig" } });
    
    const run_bench_concurrent = b.addRunArtifact(bench_concurrent);
    bench_step.dependOn(&run_bench_concurrent.step);

    const builtin = @import("builtin");
    const zig_version = builtin.zig_version;
    const min_wasm_version = std.SemanticVersion{ .major = 0, .minor = 12, .patch = 0 };
    
    const wasm_step = b.step("wasm", "Build WASM module for browser (requires Zig 0.12+)");
    
    if (zig_version.order(min_wasm_version) == .lt) {
        const skip_wasm_cmd = b.addSystemCommand(&[_][]const u8{
            "echo",
            "WASM build skipped: Requires Zig 0.12+",
        });
        wasm_step.dependOn(&skip_wasm_cmd.step);
    } else {
        const wasm_target = std.zig.CrossTarget{
            .cpu_arch = .wasm32,
            .os_tag = .freestanding,
        };
        
        const wasm_lib = b.addSharedLibrary(.{
            .name = "jaide_wasm",
            .root_source_file = .{ .path = "src/wasm/wasm_bindings.zig" },
            .target = wasm_target,
            .optimize = .ReleaseSmall,
        });
        
        wasm_lib.rdynamic = true;
        wasm_lib.addAnonymousModule("wasm_deps", .{ .source_file = .{ .path = "src/wasm_deps.zig" } });
        
        const install_wasm = b.addInstallArtifact(wasm_lib, .{});
        wasm_step.dependOn(&install_wasm.step);
        
        const copy_wasm = b.addInstallFile(
            .{ .path = "zig-out/lib/libjaide_wasm.wasm" },
            "jaide.wasm"
        );
        wasm_step.dependOn(&copy_wasm.step);
    }
}
