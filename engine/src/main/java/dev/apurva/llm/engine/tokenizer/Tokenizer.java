package dev.apurva.llm.engine.tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SentencePiece-compatible BPE tokenizer for Llama-family models.
 *
 * <p>Loads the vocabulary directly from a SentencePiece {@code tokenizer.model} (a protobuf file;
 * only the piece list is decoded). Encoding follows the classic llama2.c algorithm: split into
 * code points (with a dummy "▁" prefix and spaces mapped to "▁"), fall back to {@code <0xXX>} byte
 * tokens for unknown characters, then repeatedly merge the adjacent pair with the best score.
 */
public final class Tokenizer {
    public static final int TYPE_NORMAL = 1;
    public static final int TYPE_UNKNOWN = 2;
    public static final int TYPE_CONTROL = 3;
    public static final int TYPE_USER_DEFINED = 4;
    public static final int TYPE_UNUSED = 5;
    public static final int TYPE_BYTE = 6;

    private static final char SPACE_MARK = '\u2581'; // "▁"

    private final String[] pieces;
    private final float[] scores;
    private final int[] types;
    private final Map<String, Integer> mergeable = new HashMap<>(); // normal + user-defined pieces
    private final int[] byteToId = new int[256];
    private final byte[][] tokenBytes;
    private final int bosId;
    private final int eosId;
    private final int unkId;

    public Tokenizer(List<String> pieces, float[] scores, int[] types) {
        int n = pieces.size();
        if (scores.length != n || types.length != n) throw new IllegalArgumentException("length mismatch");
        this.pieces = pieces.toArray(new String[0]);
        this.scores = scores.clone();
        this.types = types.clone();
        java.util.Arrays.fill(byteToId, -1);

        int bos = -1, eos = -1, unk = 0;
        this.tokenBytes = new byte[n][];
        for (int i = 0; i < n; i++) {
            String p = this.pieces[i];
            int t = this.types[i];
            if (t == TYPE_NORMAL || t == TYPE_USER_DEFINED) {
                mergeable.putIfAbsent(p, i);
                tokenBytes[i] = p.replace(SPACE_MARK, ' ').getBytes(StandardCharsets.UTF_8);
            } else if (t == TYPE_BYTE) {
                int b = parseByteToken(p);
                if (b >= 0) byteToId[b] = i;
                tokenBytes[i] = b >= 0 ? new byte[] {(byte) b} : new byte[0];
            } else {
                tokenBytes[i] = new byte[0]; // control / unknown / unused emit nothing
                if (t == TYPE_UNKNOWN) unk = i;
            }
            if (p.equals("<s>") && t == TYPE_CONTROL) bos = i;
            if (p.equals("</s>") && t == TYPE_CONTROL) eos = i;
        }
        this.bosId = bos >= 0 ? bos : 1;
        this.eosId = eos >= 0 ? eos : 2;
        this.unkId = unk;
    }

    private static int parseByteToken(String p) {
        if (p.length() == 6 && p.startsWith("<0x") && p.endsWith(">")) {
            try {
                return Integer.parseInt(p.substring(3, 5), 16);
            } catch (NumberFormatException ignored) {
                return -1;
            }
        }
        return -1;
    }

    public int vocabSize() { return pieces.length; }
    public int bosId() { return bosId; }
    public int eosId() { return eosId; }
    public String piece(int id) { return pieces[id]; }

    // ---- loading -----------------------------------------------------------

    public static Tokenizer load(Path modelDir) throws IOException {
        Path p = modelDir.resolve("tokenizer.model");
        if (!Files.exists(p)) {
            throw new IOException("Missing " + p + " (a SentencePiece tokenizer.model is required)");
        }
        return fromSentencePiece(Files.readAllBytes(p));
    }

    public static Tokenizer fromSentencePiece(byte[] data) {
        List<String> pieces = new ArrayList<>();
        List<Float> scores = new ArrayList<>();
        List<Integer> types = new ArrayList<>();

        Reader r = new Reader(data, 0, data.length);
        while (r.hasMore()) {
            long tag = r.varint();
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            if (field == 1 && wire == 2) {
                int len = (int) r.varint();
                Reader sub = new Reader(data, r.pos, r.pos + len);
                r.pos += len;
                String piece = "";
                float score = 0f;
                int type = TYPE_NORMAL;
                while (sub.hasMore()) {
                    long t2 = sub.varint();
                    int f2 = (int) (t2 >>> 3);
                    int w2 = (int) (t2 & 7);
                    if (f2 == 1 && w2 == 2) {
                        int l = (int) sub.varint();
                        piece = new String(data, sub.pos, l, StandardCharsets.UTF_8);
                        sub.pos += l;
                    } else if (f2 == 2 && w2 == 5) {
                        score = Float.intBitsToFloat(sub.fixed32());
                    } else if (f2 == 3 && w2 == 0) {
                        type = (int) sub.varint();
                    } else {
                        sub.skip(w2);
                    }
                }
                pieces.add(piece);
                scores.add(score);
                types.add(type);
            } else {
                r.skip(wire);
            }
        }
        if (pieces.isEmpty()) throw new IllegalArgumentException("No pieces found in SentencePiece model");
        float[] sc = new float[scores.size()];
        int[] ty = new int[types.size()];
        for (int i = 0; i < sc.length; i++) {
            sc[i] = scores.get(i);
            ty[i] = types.get(i);
        }
        return new Tokenizer(pieces, sc, ty);
    }

    private static final class Reader {
        final byte[] b;
        int pos;
        final int end;

        Reader(byte[] b, int pos, int end) {
            this.b = b;
            this.pos = pos;
            this.end = end;
        }

        boolean hasMore() { return pos < end; }

        long varint() {
            long v = 0;
            int shift = 0;
            while (true) {
                if (pos >= end) throw new IllegalArgumentException("Truncated protobuf varint");
                int c = b[pos++] & 0xFF;
                v |= (long) (c & 0x7F) << shift;
                if ((c & 0x80) == 0) return v;
                shift += 7;
                if (shift > 63) throw new IllegalArgumentException("Bad protobuf varint");
            }
        }

        int fixed32() {
            if (pos + 4 > end) throw new IllegalArgumentException("Truncated protobuf fixed32");
            int v = (b[pos] & 0xFF) | (b[pos + 1] & 0xFF) << 8 | (b[pos + 2] & 0xFF) << 16 | (b[pos + 3] & 0xFF) << 24;
            pos += 4;
            return v;
        }

        void skip(int wire) {
            switch (wire) {
                case 0 -> varint();
                case 1 -> pos += 8;
                case 2 -> {
                    int len = (int) varint(); // read first: varint() advances pos
                    pos += len;
                }
                case 5 -> pos += 4;
                default -> throw new IllegalArgumentException("Unsupported protobuf wire type " + wire);
            }
            if (pos > end) throw new IllegalArgumentException("Truncated protobuf field");
        }
    }

    // ---- encode ------------------------------------------------------------

    public int[] encode(String text, boolean addBos, boolean addEos) {
        int[] ids = new int[text.length() * 4 + 4]; // worst case: 4 byte tokens per UTF-16 unit
        int n = 0;
        if (addBos) ids[n++] = bosId;

        if (!text.isEmpty()) {
            String s = SPACE_MARK + text.replace(' ', SPACE_MARK);
            for (int i = 0; i < s.length(); ) {
                int cp = s.codePointAt(i);
                int len = Character.charCount(cp);
                String ch = s.substring(i, i + len);
                Integer id = mergeable.get(ch);
                if (id != null) {
                    ids[n++] = id;
                } else {
                    for (byte b : ch.getBytes(StandardCharsets.UTF_8)) {
                        int bid = byteToId[b & 0xFF];
                        ids[n++] = bid >= 0 ? bid : unkId;
                    }
                }
                i += len;
            }

            // Greedy BPE: merge the best-scoring adjacent pair until none remain.
            int start = addBos ? 1 : 0;
            while (true) {
                float best = Float.NEGATIVE_INFINITY;
                int bestIdx = -1, bestId = -1;
                for (int i = start; i < n - 1; i++) {
                    Integer id = mergeable.get(pieces[ids[i]] + pieces[ids[i + 1]]);
                    if (id != null && scores[id] > best) {
                        best = scores[id];
                        bestIdx = i;
                        bestId = id;
                    }
                }
                if (bestIdx < 0) break;
                ids[bestIdx] = bestId;
                System.arraycopy(ids, bestIdx + 2, ids, bestIdx + 1, n - bestIdx - 2);
                n--;
            }
        }

        if (addEos) ids[n++] = eosId;
        return java.util.Arrays.copyOf(ids, n);
    }

    // ---- decode ------------------------------------------------------------

    /** Raw bytes for one token (empty for control tokens). Multi-byte characters may span tokens. */
    public byte[] decodeToken(int id) {
        if (id < 0 || id >= tokenBytes.length) throw new IllegalArgumentException("bad token id " + id);
        return tokenBytes[id];
    }

    /** Decodes a whole sequence; drops the dummy-prefix space SentencePiece adds at the start. */
    public String decode(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (int id : ids) out.writeBytes(decodeToken(id));
        String s = out.toString(StandardCharsets.UTF_8);
        return s.startsWith(" ") ? s.substring(1) : s;
    }
}
