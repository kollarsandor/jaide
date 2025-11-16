const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const Error = types.Error;
const testing = std.testing;

pub const MGT = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    prefixes: std.StringHashMap(u32),
    suffixes: std.StringHashMap(u32),
    roots: std.StringHashMap(u32),
    bpe_pairs: std.StringHashMap(BPEMerge),
    anchors: std.StringHashMap(u64),
    allocated_strings: std.ArrayList([]u8),
    allocator: Allocator,
    vocab_size: u32,
    next_token_id: u32,

    const BPEMerge = struct {
        token_id: u32,
        priority: u32,
    };

    const SPECIAL_TOKENS = struct {
        const PAD: u32 = 0;
        const UNK: u32 = 1;
        const BOS: u32 = 2;
        const EOS: u32 = 3;
    };

    pub fn init(allocator: Allocator, vocab: []const []const u8, anchors: []const []const u8) !MGT {
        var token_to_id = std.StringHashMap(u32).init(allocator);
        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        var prefixes = std.StringHashMap(u32).init(allocator);
        var suffixes = std.StringHashMap(u32).init(allocator);
        var roots = std.StringHashMap(u32).init(allocator);
        var bpe_pairs = std.StringHashMap(BPEMerge).init(allocator);
        var anch_map = std.StringHashMap(u64).init(allocator);
        var allocated = std.ArrayList([]u8).init(allocator);

        const special = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };
        var i: usize = 0;
        for (special) |tok| {
            const tok_copy = try allocator.dupe(u8, tok);
            try allocated.append(tok_copy);
            try token_to_id.put(tok_copy, @intCast(i));
            try id_to_token.put(@intCast(i), tok_copy);
            i += 1;
        }

        var next_id: u32 = 4;
        for (vocab) |word| {
            if (!token_to_id.contains(word)) {
                const word_copy = try allocator.dupe(u8, word);
                try allocated.append(word_copy);
                try token_to_id.put(word_copy, next_id);
                try id_to_token.put(next_id, word_copy);
                next_id += 1;
            }
        }

        var mgt = MGT{
            .token_to_id = token_to_id,
            .id_to_token = id_to_token,
            .prefixes = prefixes,
            .suffixes = suffixes,
            .roots = roots,
            .bpe_pairs = bpe_pairs,
            .anchors = anch_map,
            .allocated_strings = allocated,
            .allocator = allocator,
            .vocab_size = 50000,
            .next_token_id = next_id,
        };

        try mgt.initMorphemes();

        for (anchors) |anch| {
            if (mgt.token_to_id.get(anch)) |tid| {
                const h: u64 = @intCast(tid);
                const anch_copy = try allocator.dupe(u8, anch);
                try mgt.allocated_strings.append(anch_copy);
                try mgt.anchors.put(anch_copy, h);
            }
        }

        return mgt;
    }

    fn initMorphemes(self: *MGT) !void {
        const prefix_list = [_][]const u8{
            "un",  "re",   "pre",  "dis",  "mis",  "over", "under", "out",
            "sub", "inter", "fore", "de",   "trans", "super", "semi", "anti",
            "mid", "non",  "ex",   "post", "pro",  "co",   "en",   "em",
        };

        for (prefix_list) |prefix| {
            if (!self.token_to_id.contains(prefix)) {
                const prefix_copy = try self.allocator.dupe(u8, prefix);
                try self.allocated_strings.append(prefix_copy);
                try self.token_to_id.put(prefix_copy, self.next_token_id);
                try self.id_to_token.put(self.next_token_id, prefix_copy);
                try self.prefixes.put(prefix_copy, self.next_token_id);
                self.next_token_id += 1;
            }
        }

        const suffix_list = [_][]const u8{
            "ing", "ed",  "er",   "est",  "ly",   "tion", "sion", "ness",
            "ment", "ful", "less", "ous",  "ive",  "able", "ible", "al",
            "ial", "y",   "s",    "es",   "en",   "ize",  "ise",  "ate",
        };

        for (suffix_list) |suffix| {
            if (!self.token_to_id.contains(suffix)) {
                const suffix_copy = try self.allocator.dupe(u8, suffix);
                try self.allocated_strings.append(suffix_copy);
                try self.token_to_id.put(suffix_copy, self.next_token_id);
                try self.id_to_token.put(self.next_token_id, suffix_copy);
                try self.suffixes.put(suffix_copy, self.next_token_id);
                self.next_token_id += 1;
            }
        }
    }

    pub fn deinit(self: *MGT) void {
        self.token_to_id.deinit();
        self.id_to_token.deinit();
        self.prefixes.deinit();
        self.suffixes.deinit();
        self.roots.deinit();
        self.bpe_pairs.deinit();
        self.anchors.deinit();
        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }
        self.allocated_strings.deinit();
    }

    pub fn encode(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < text.len) {
            while (i < text.len and (text[i] == ' ' or text[i] == '\n' or text[i] == '\t' or text[i] == '\r')) {
                i += 1;
            }
            if (i >= text.len) break;
            
            var word_end = i;
            while (word_end < text.len) {
                const c = text[word_end];
                if (c == ' ' or c == '\n' or c == '\t' or c == '\r' or c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or c == ':' or c == '"' or c == '\'' or c == '(' or c == ')' or c == '[' or c == ']' or c == '{' or c == '}') {
                    break;
                }
                word_end += 1;
            }
            
            if (word_end > i) {
                const word = text[i..word_end];
                if (self.token_to_id.get(word)) |tid| {
                    try out_tokens.append(tid);
                } else {
                    if (self.next_token_id < self.vocab_size) {
                        const word_copy = try self.allocator.dupe(u8, word);
                        try self.allocated_strings.append(word_copy);
                        try self.token_to_id.put(word_copy, self.next_token_id);
                        try self.id_to_token.put(self.next_token_id, word_copy);
                        try out_tokens.append(self.next_token_id);
                        self.next_token_id += 1;
                    } else {
                        try out_tokens.append(SPECIAL_TOKENS.UNK);
                    }
                }
                i = word_end;
            }
            
            if (i < text.len) {
                const punct = text[i];
                if (punct == '.' or punct == ',' or punct == '!' or punct == '?' or punct == ';' or punct == ':' or punct == '"' or punct == '\'' or punct == '(' or punct == ')' or punct == '[' or punct == ']' or punct == '{' or punct == '}') {
                    const punct_str = text[i..i+1];
                    if (self.token_to_id.get(punct_str)) |tid| {
                        try out_tokens.append(tid);
                    } else {
                        if (self.next_token_id < self.vocab_size) {
                            const punct_copy = try self.allocator.dupe(u8, punct_str);
                            try self.allocated_strings.append(punct_copy);
                            try self.token_to_id.put(punct_copy, self.next_token_id);
                            try self.id_to_token.put(self.next_token_id, punct_copy);
                            try out_tokens.append(self.next_token_id);
                            self.next_token_id += 1;
                        } else {
                            try out_tokens.append(SPECIAL_TOKENS.UNK);
                        }
                    }
                    i += 1;
                }
            }
        }
    }

    fn morphDecompose(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !?usize {
        if (text.len == 0) return null;

        var word_end: usize = 0;
        while (word_end < text.len) : (word_end += 1) {
            const c = text[word_end];
            if (c == ' ' or c == '\n' or c == '\t' or c == '.' or c == ',' or c == '!' or c == '?') {
                break;
            }
        }
        if (word_end == 0) return null;

        const word = text[0..word_end];

        var prefix_len: usize = 0;
        var best_prefix: ?[]const u8 = null;
        var prefix_it = self.prefixes.iterator();
        while (prefix_it.next()) |entry| {
            const prefix = entry.key_ptr.*;
            if (word.len > prefix.len and mem.startsWith(u8, word, prefix)) {
                if (prefix.len > prefix_len) {
                    prefix_len = prefix.len;
                    best_prefix = prefix;
                }
            }
        }

        var suffix_len: usize = 0;
        var best_suffix: ?[]const u8 = null;
        var suffix_it = self.suffixes.iterator();
        while (suffix_it.next()) |entry| {
            const suffix = entry.key_ptr.*;
            if (word.len > suffix.len and mem.endsWith(u8, word, suffix)) {
                if (suffix.len > suffix_len) {
                    suffix_len = suffix.len;
                    best_suffix = suffix;
                }
            }
        }

        if (prefix_len > 0 or suffix_len > 0) {
            if (best_prefix) |prefix| {
                if (self.token_to_id.get(prefix)) |tid| {
                    try out_tokens.append(tid);
                }
            }

            const root_start = prefix_len;
            const root_end = word.len - suffix_len;
            if (root_end > root_start) {
                const root = word[root_start..root_end];
                if (self.token_to_id.get(root)) |tid| {
                    try out_tokens.append(tid);
                } else {
                    const root_id = try self.addToken(root);
                    const root_str = self.id_to_token.get(root_id).?;
                    try self.roots.put(root_str, root_id);
                    try out_tokens.append(root_id);
                }
            }

            if (best_suffix) |suffix| {
                if (self.token_to_id.get(suffix)) |tid| {
                    try out_tokens.append(tid);
                }
            }

            return word_end;
        }

        return null;
    }

    fn addToken(self: *MGT, token: []const u8) !u32 {
        if (self.token_to_id.get(token)) |existing| {
            return existing;
        }

        if (self.next_token_id >= self.vocab_size) {
            return SPECIAL_TOKENS.UNK;
        }

        const token_copy = try self.allocator.dupe(u8, token);
        try self.allocated_strings.append(token_copy);
        try self.token_to_id.put(token_copy, self.next_token_id);
        try self.id_to_token.put(self.next_token_id, token_copy);
        const id = self.next_token_id;
        self.next_token_id += 1;
        return id;
    }

    fn encodeBPE(self: *MGT, text: []const u8) ![]u32 {
        if (text.len == 0) return &.{};

        var tokens = std.ArrayList(u32).init(self.allocator);
        var byte_tokens = std.ArrayList([]u8).init(self.allocator);
        defer {
            for (byte_tokens.items) |bt| {
                self.allocator.free(bt);
            }
            byte_tokens.deinit();
        }

        for (text) |byte| {
            const byte_str = try std.fmt.allocPrint(self.allocator, "<{x:0>2}>", .{byte});
            try byte_tokens.append(byte_str);
        }

        while (byte_tokens.items.len > 1) {
            var best_priority: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;

            var i: usize = 0;
            while (i + 1 < byte_tokens.items.len) : (i += 1) {
                const pair = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}{s}",
                    .{ byte_tokens.items[i], byte_tokens.items[i + 1] },
                );
                defer self.allocator.free(pair);

                if (self.bpe_pairs.get(pair)) |merge| {
                    if (merge.priority < best_priority) {
                        best_priority = merge.priority;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == null) break;

            const idx = best_idx.?;
            const merged = try std.fmt.allocPrint(
                self.allocator,
                "{s}{s}",
                .{ byte_tokens.items[idx], byte_tokens.items[idx + 1] },
            );

            self.allocator.free(byte_tokens.items[idx]);
            self.allocator.free(byte_tokens.items[idx + 1]);

            byte_tokens.items[idx] = merged;
            _ = byte_tokens.orderedRemove(idx + 1);
        }

        for (byte_tokens.items) |bt| {
            if (self.token_to_id.get(bt)) |tid| {
                try tokens.append(tid);
            } else {
                const tid = try self.addToken(bt);
                try tokens.append(tid);
            }
        }

        return try tokens.toOwnedSlice();
    }

    pub fn trainBPE(self: *MGT, corpus: []const []const u8, num_merges: u32) !void {
        var pair_freqs = std.StringHashMap(u32).init(self.allocator);
        defer {
            var it = pair_freqs.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
            }
            pair_freqs.deinit();
        }

        for (corpus) |text| {
            var i: usize = 0;
            while (i + 1 < text.len) : (i += 1) {
                const pair = try std.fmt.allocPrint(
                    self.allocator,
                    "<{x:0>2}><{x:0>2}>",
                    .{ text[i], text[i + 1] },
                );

                if (pair_freqs.getPtr(pair)) |freq| {
                    freq.* += 1;
                    self.allocator.free(pair);
                } else {
                    try pair_freqs.put(pair, 1);
                }
            }
        }

        var merge_count: u32 = 0;
        while (merge_count < num_merges) : (merge_count += 1) {
            var max_freq: u32 = 0;
            var best_pair: ?[]const u8 = null;

            var it = pair_freqs.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.* > max_freq) {
                    max_freq = entry.value_ptr.*;
                    best_pair = entry.key_ptr.*;
                }
            }

            if (best_pair == null or max_freq < 2) break;

            const pair_copy = try self.allocator.dupe(u8, best_pair.?);
            try self.allocated_strings.append(pair_copy);

            const merge_token_id = try self.addToken(pair_copy);
            try self.bpe_pairs.put(pair_copy, .{
                .token_id = merge_token_id,
                .priority = merge_count,
            });

            if (pair_freqs.fetchRemove(best_pair.?)) |removed| {
                self.allocator.free(removed.key);
            }
        }
    }

    pub fn decode(self: *MGT, tokens: []const u32, out_text: *std.ArrayList(u8)) !void {
        for (tokens) |tok| {
            if (self.id_to_token.get(tok)) |token_str| {
                if (mem.startsWith(u8, token_str, "<") and mem.endsWith(u8, token_str, ">")) {
                    const hex = token_str[1 .. token_str.len - 1];
                    const byte = try std.fmt.parseInt(u8, hex, 16);
                    try out_text.append(byte);
                } else if (mem.eql(u8, token_str, "[PAD]") or
                    mem.eql(u8, token_str, "[UNK]") or
                    mem.eql(u8, token_str, "[BOS]") or
                    mem.eql(u8, token_str, "[EOS]"))
                {
                    continue;
                } else {
                    try out_text.appendSlice(token_str);
                }
            } else {
                try out_text.appendSlice("[UNK]");
            }
        }
    }

    pub fn longestMatch(self: *MGT, text: []const u8, start: usize) usize {
        var max_len: usize = 0;
        var len: usize = 1;

        while (start + len <= text.len) : (len += 1) {
            const substr = text[start .. start + len];
            if (self.token_to_id.contains(substr)) {
                max_len = len;
            }
        }

        return max_len;
    }

    pub fn vocabSize(self: *const MGT) usize {
        return self.token_to_id.count();
    }

    pub fn addVocabWord(self: *MGT, word: []const u8, is_anchor: bool) !void {
        _ = try self.addToken(word);
        if (is_anchor) {
            const h: u64 = @intCast(self.token_to_id.get(word).?);
            try self.anchors.put(word, h);
        }
    }

    pub fn removeVocabWord(self: *MGT, word: []const u8) void {
        _ = self.token_to_id.remove(word);
        _ = self.anchors.remove(word);
        _ = self.prefixes.remove(word);
        _ = self.suffixes.remove(word);
        _ = self.roots.remove(word);
    }

    pub fn tokenizeWithAnchors(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: *std.ArrayList(usize)) !void {
        var i: usize = 0;
        while (i < text.len) {
            const match_len = self.longestMatch(text, i);
            if (match_len > 0) {
                const word = text[i .. i + match_len];
                if (self.token_to_id.get(word)) |tid| {
                    try out_tokens.append(tid);
                    if (self.anchors.contains(word)) {
                        try out_anchors.append(i);
                    }
                    i += match_len;
                    continue;
                }
            }

            const bpe_tokens = try self.encodeBPE(text[i..i+1]);
            defer self.allocator.free(bpe_tokens);
            for (bpe_tokens) |tok| {
                try out_tokens.append(tok);
            }
            i += 1;
        }
    }

    pub fn detokenize(self: *MGT, tokens: []const u32) ![]u8 {
        var text = std.ArrayList(u8).init(self.allocator);
        try self.decode(tokens, &text);
        return try text.toOwnedSlice();
    }

    pub fn encodeBatch(self: *MGT, texts: []const []const u8, allocator: Allocator) ![][]u32 {
        const results = try allocator.alloc([]u32, texts.len);
        var i: usize = 0;
        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            try self.encode(text, &tokens);
            results[i] = try tokens.toOwnedSlice();
            i += 1;
        }
        return results;
    }

    pub fn batchDetokenize(self: *MGT, token_lists: []const []const u32, allocator: Allocator) ![][]u8 {
        const results = try allocator.alloc([]u8, token_lists.len);
        var i: usize = 0;
        for (token_lists) |tokens| {
            results[i] = try self.detokenize(tokens);
            i += 1;
        }
        return results;
    }

    pub fn saveVocab(self: *MGT, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();

        const size = self.vocabSize();
        try writer.writeInt(u32, @as(u32, @intCast(size)), .Little);

        var it = self.token_to_id.iterator();
        while (it.next()) |entry| {
            const word = entry.key_ptr.*;
            const id = entry.value_ptr.*;
            try writer.writeInt(u32, @as(u32, @intCast(word.len)), .Little);
            try writer.writeAll(word);
            try writer.writeInt(u32, id, .Little);
        }
    }

    pub fn loadVocab(self: *MGT, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();

        const size = try reader.readInt(u32, .Little);
        var i: usize = 0;
        while (i < size) : (i += 1) {
            const word_len = try reader.readInt(u32, .Little);
            var word_buf = try self.allocator.alloc(u8, word_len);
            defer self.allocator.free(word_buf);
            try reader.readNoEof(word_buf);
            const id = try reader.readInt(u32, .Little);

            const word_copy = try self.allocator.dupe(u8, word_buf);
            try self.allocated_strings.append(word_copy);
            try self.token_to_id.put(word_copy, id);
            try self.id_to_token.put(id, word_copy);

            if (id >= self.next_token_id) {
                self.next_token_id = id + 1;
            }
        }
    }

    pub fn unknownReplacement(self: *MGT, context: []const u8) u32 {
        _ = self;
        _ = context;
        return SPECIAL_TOKENS.UNK;
    }

    pub fn subwordSplit(self: *MGT, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        var i: usize = 0;
        while (i < word.len) {
            const match = self.longestMatch(word, i);
            if (match > 0) {
                const found_word = word[i .. i + match];
                if (self.token_to_id.get(found_word)) |tid| {
                    try tokens.append(tid);
                    i += match;
                    continue;
                }
            }

            const bpe_tokens = try self.encodeBPE(word[i..i+1]);
            defer self.allocator.free(bpe_tokens);
            for (bpe_tokens) |tok| {
                try tokens.append(tok);
            }
            i += 1;
        }
        return try tokens.toOwnedSlice();
    }

    pub fn mergeSubwords(self: *MGT, subwords: []const []const u32) ![]u32 {
        var merged = std.ArrayList(u32).init(self.allocator);
        for (subwords) |sw| {
            for (sw) |tok| {
                try merged.append(tok);
            }
        }
        return try merged.toOwnedSlice();
    }

    pub fn validateTokens(self: *MGT, tokens: []const u32) bool {
        const max_token = self.vocab_size;
        for (tokens) |tok| {
            if (tok > max_token) return false;
        }
        return true;
    }

    pub fn coverage(self: *MGT, corpus: []const u8) f32 {
        var covered: usize = 0;
        var i: usize = 0;
        while (i < corpus.len) {
            const m = self.longestMatch(corpus, i);
            if (m > 0) {
                covered += m;
                i += m;
            } else {
                i += 1;
            }
        }
        if (corpus.len == 0) return 0.0;
        return @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(corpus.len));
    }
};

test "MGT encode decode" {
    var gpa = testing.allocator;
    const vocab = &.{ "hello", "world", " " };
    const anchors = &.{"hello"};
    var mgt = try MGT.init(gpa, vocab, anchors);
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    try mgt.encode("hello world", &tokens);
    try testing.expect(tokens.items.len >= 2);
    var text = std.ArrayList(u8).init(gpa);
    defer text.deinit();
    try mgt.decode(tokens.items, &text);
}

test "MGT add remove vocab" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    try mgt.addVocabWord("test", true);
    try testing.expect(mgt.anchors.contains("test"));
    mgt.removeVocabWord("test");
    try testing.expect(!mgt.anchors.contains("test"));
}

test "MGT longest match" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hello", "hell" }, &.{});
    defer mgt.deinit();
    const len = mgt.longestMatch("hello", 0);
    try testing.expectEqual(@as(usize, 5), len);
}

test "MGT batch encode" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{});
    defer mgt.deinit();
    const texts = &.{ "a", "b" };
    const batches = try mgt.encodeBatch(texts, gpa);
    defer {
        for (batches) |batch| {
            gpa.free(batch);
        }
        gpa.free(batches);
    }
    try testing.expect(batches.len == 2);
}

test "MGT subword split" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hel", "lo" }, &.{});
    defer mgt.deinit();
    const sub = try mgt.subwordSplit("hello");
    defer gpa.free(sub);
    try testing.expect(sub.len >= 1);
}

test "MGT coverage" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hello", "world" }, &.{});
    defer mgt.deinit();
    const cov = mgt.coverage("hello world");
    try testing.expect(cov > 0.0);
}

test "MGT validate" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{"a"}, &.{});
    defer mgt.deinit();
    const valid = mgt.validateTokens(&.{0});
    try testing.expect(valid);
}

test "MGT tokenize with anchors" {
    var gpa = testing.allocator;
    const vocab = &.{ "test", "anchor" };
    const anchors = &.{"anchor"};
    var mgt = try MGT.init(gpa, vocab, anchors);
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    var anchor_positions = std.ArrayList(usize).init(gpa);
    defer anchor_positions.deinit();
    try mgt.tokenizeWithAnchors("testanchor", &tokens, &anchor_positions);
    try testing.expect(tokens.items.len >= 1);
}

test "MGT batch detokenize" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{});
    defer mgt.deinit();
    const token_lists = &[_][]const u32{
        &.{4},
        &.{5},
    };
    const results = try mgt.batchDetokenize(token_lists, gpa);
    defer {
        for (results) |result| {
            gpa.free(result);
        }
        gpa.free(results);
    }
    try testing.expect(results.len == 2);
}

test "MGT vocab size" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b", "c" }, &.{});
    defer mgt.deinit();
    const size = mgt.vocabSize();
    try testing.expect(size >= 3);
}

test "MGT save and load vocab" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "test", "vocab" }, &.{});
    defer mgt.deinit();
    try mgt.saveVocab("test_vocab.bin");
    defer {
        std.fs.cwd().deleteFile("test_vocab.bin") catch |err| {
            std.log.warn("Failed to delete test file: {}", .{err});
        };
    }
    var mgt2 = try MGT.init(gpa, &.{}, &.{});
    defer mgt2.deinit();
    try mgt2.loadVocab("test_vocab.bin");
    try testing.expect(mgt2.vocabSize() >= 1);
}

test "MGT merge subwords" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const sub1 = &[_]u32{ 1, 2 };
    const sub2 = &[_]u32{ 3, 4 };
    const subwords = &[_][]const u32{ sub1, sub2 };
    const merged = try mgt.mergeSubwords(subwords);
    defer gpa.free(merged);
    try testing.expectEqual(@as(usize, 4), merged.len);
}

test "MGT unknown replacement" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const replacement = mgt.unknownReplacement("context");
    try testing.expectEqual(MGT.SPECIAL_TOKENS.UNK, replacement);
}

test "MGT morphological decomposition" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "run", "walk" }, &.{});
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    try mgt.encode("running", &tokens);
    try testing.expect(tokens.items.len >= 2);
}

test "MGT BPE training" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const corpus = &.{ "hello", "help", "held" };
    try mgt.trainBPE(corpus, 10);
    try testing.expect(mgt.bpe_pairs.count() > 0);
}

test "MGT deterministic encoding" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "test", "data" }, &.{});
    defer mgt.deinit();

    var tokens1 = std.ArrayList(u32).init(gpa);
    defer tokens1.deinit();
    try mgt.encode("test data", &tokens1);

    var tokens2 = std.ArrayList(u32).init(gpa);
    defer tokens2.deinit();
    try mgt.encode("test data", &tokens2);

    try testing.expectEqualSlices(u32, tokens1.items, tokens2.items);
}
