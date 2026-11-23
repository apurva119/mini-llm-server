package dev.apurva.llm.engine;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/** Test fixtures: a tiny random Llama-style model, a safetensors writer and a SentencePiece encoder. */
final class TestModels {
    private TestModels() {}

    static final int HIDDEN = 16, INTER = 32, HEADS = 4, KV_HEADS = 2, LAYERS = 2, MAX_SEQ = 64;
    static final int VOCAB = 3 + 256 + 8; // unk, bos, eos, 256 bytes, 8 normal pieces

    static final String CONFIG = """
            {"hidden_size": 16, "intermediate_size": 32, "vocab_size": %d,
             "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 2,
             "max_position_embeddings": 64, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
             "hidden_act": "silu", "tie_word_embeddings": true, "bos_token_id": 1, "eos_token_id": 2}
            """.formatted(VOCAB);

    /** Writes config.json, model.safetensors (tied embeddings) and tokenizer.model into {@code dir}. */
    static void writeModelDir(Path dir, long seed) throws IOException {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("config.json"), CONFIG);

        Random rnd = new Random(seed);
        Map<String, float[]> data = new LinkedHashMap<>();
        Map<String, int[]> shapes = new LinkedHashMap<>();
        put(data, shapes, "model.embed_tokens.weight", rnd, 0.3f, VOCAB, HIDDEN);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            put(data, shapes, p + "self_attn.q_proj.weight", rnd, 0.3f, HEADS * 4, HIDDEN);
            put(data, shapes, p + "self_attn.k_proj.weight", rnd, 0.3f, KV_HEADS * 4, HIDDEN);
            put(data, shapes, p + "self_attn.v_proj.weight", rnd, 0.3f, KV_HEADS * 4, HIDDEN);
            put(data, shapes, p + "self_attn.o_proj.weight", rnd, 0.3f, HIDDEN, HEADS * 4);
            put(data, shapes, p + "mlp.gate_proj.weight", rnd, 0.3f, INTER, HIDDEN);
            put(data, shapes, p + "mlp.up_proj.weight", rnd, 0.3f, INTER, HIDDEN);
            put(data, shapes, p + "mlp.down_proj.weight", rnd, 0.3f, HIDDEN, INTER);
            ones(data, shapes, p + "input_layernorm.weight", HIDDEN);
            ones(data, shapes, p + "post_attention_layernorm.weight", HIDDEN);
        }
        ones(data, shapes, "model.norm.weight", HIDDEN);
        writeF32(dir.resolve("model.safetensors"), data, shapes);

        Files.write(dir.resolve("tokenizer.model"), tokenizerBytes());
    }

    private static void put(Map<String, float[]> d, Map<String, int[]> s, String name, Random r, float std, int... shape) {
        int n = 1;
        for (int x : shape) n *= x;
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) r.nextGaussian() * std;
        d.put(name, a);
        s.put(name, shape);
    }

    private static void ones(Map<String, float[]> d, Map<String, int[]> s, String name, int n) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, 1f);
        d.put(name, a);
        s.put(name, new int[] {n});
    }

    static void writeF32(Path file, Map<String, float[]> data, Map<String, int[]> shapes) throws IOException {
        Map<String, String> dtypes = new LinkedHashMap<>();
        Map<String, byte[]> raw = new LinkedHashMap<>();
        for (var e : data.entrySet()) {
            ByteBuffer bb = ByteBuffer.allocate(e.getValue().length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float f : e.getValue()) bb.putFloat(f);
            raw.put(e.getKey(), bb.array());
            dtypes.put(e.getKey(), "F32");
        }
        writeRaw(file, raw, dtypes, shapes);
    }

    static void writeRaw(Path file, Map<String, byte[]> raw, Map<String, String> dtypes, Map<String, int[]> shapes)
            throws IOException {
        StringBuilder h = new StringBuilder("{");
        long off = 0;
        boolean first = true;
        for (var e : raw.entrySet()) {
            if (!first) h.append(',');
            first = false;
            h.append('"').append(e.getKey()).append("\":{\"dtype\":\"").append(dtypes.get(e.getKey()))
                    .append("\",\"shape\":").append(java.util.Arrays.toString(shapes.get(e.getKey())))
                    .append(",\"data_offsets\":[").append(off).append(',').append(off + e.getValue().length).append("]}");
            off += e.getValue().length;
        }
        h.append('}');
        byte[] hb = h.toString().getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(hb.length).array());
        out.writeBytes(hb);
        for (byte[] b : raw.values()) out.writeBytes(b);
        Files.write(file, out.toByteArray());
    }

    // ---- SentencePiece model encoder ---------------------------------------------------------

    static final String[] NORMAL = {"\u2581", "a", "b", "c", "\u2581a", "ab", "\u2581ab", "abc"};
    static final float[] NORMAL_SCORES = {-10, -10, -10, -10, -3, -2, -1, -1.5f};

    static byte[] tokenizerBytes() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(new byte[] {0x12, 2, 0x08, 0x01}); // unrelated field 2 (must be skipped)
        piece(out, "<unk>", 0f, 2);
        piece(out, "<s>", 0f, 3);
        piece(out, "</s>", 0f, 3);
        for (int b = 0; b < 256; b++) piece(out, String.format("<0x%02X>", b), 0f, 6);
        for (int i = 0; i < NORMAL.length; i++) piece(out, NORMAL[i], NORMAL_SCORES[i], 1);
        return out.toByteArray();
    }

    private static void piece(ByteArrayOutputStream out, String p, float score, int type) {
        byte[] pb = p.getBytes(StandardCharsets.UTF_8);
        ByteArrayOutputStream sub = new ByteArrayOutputStream();
        sub.write(0x0A);
        sub.write(pb.length);
        sub.writeBytes(pb);
        sub.write(0x15);
        sub.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(score).array());
        sub.write(0x18); // field 3 (type), varint
        sub.write(type);
        out.write(0x0A);
        out.write(sub.size());
        out.writeBytes(sub.toByteArray());
    }

    static List<Integer> ids(int... a) {
        List<Integer> l = new ArrayList<>();
        for (int x : a) l.add(x);
        return l;
    }
}
