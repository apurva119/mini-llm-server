package dev.apurva.llm.engine.sampling;

import java.util.Arrays;
import java.util.Random;

/**
 * Chooses the next token from logits: greedy (temperature 0), or temperature scaling with optional
 * top-k and top-p (nucleus) filtering. Not thread-safe; use one instance per generation.
 */
public final class Sampler {
    private final float temperature;
    private final int topK;
    private final float topP;
    private final Random rng;

    /**
     * @param temperature 0 means greedy argmax
     * @param topK        0 disables top-k
     * @param topP        1 disables top-p; must be in (0, 1]
     */
    public Sampler(float temperature, int topK, float topP, long seed) {
        if (temperature < 0 || Float.isNaN(temperature)) throw new IllegalArgumentException("temperature must be >= 0");
        if (topK < 0) throw new IllegalArgumentException("topK must be >= 0");
        if (!(topP > 0f && topP <= 1f)) throw new IllegalArgumentException("topP must be in (0, 1]");
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.rng = new Random(seed);
    }

    public static Sampler greedy() {
        return new Sampler(0f, 0, 1f, 0L);
    }

    public int sample(float[] logits) {
        int n = logits.length;
        if (temperature == 0f) return argmax(logits);

        boolean filter = (topK > 0 && topK < n) || topP < 1f;
        if (!filter) {
            float[] p = new float[n];
            probabilities(logits, null, p);
            return draw(p, null, n);
        }

        // Sort token indices by logit, descending, using order-preserving long keys.
        long[] keys = new long[n];
        for (int i = 0; i < n; i++) {
            int bits = Float.floatToIntBits(logits[i]);
            if (bits < 0) bits ^= 0x7FFFFFFF;
            keys[i] = ((long) bits << 32) | i;
        }
        Arrays.sort(keys);

        int limit = (topK > 0 && topK < n) ? topK : n;
        int[] idx = new int[limit];
        for (int i = 0; i < limit; i++) idx[i] = (int) (keys[n - 1 - i] & 0xFFFFFFFFL);

        float[] p = new float[limit];
        probabilities(logits, idx, p);

        int cut = limit;
        if (topP < 1f) {
            float cum = 0f;
            for (int i = 0; i < limit; i++) {
                cum += p[i];
                if (cum >= topP) { cut = i + 1; break; }
            }
        }
        return draw(p, idx, cut);
    }

    /** Softmax over temperature-scaled logits; {@code idx} (if non-null) selects/orders the entries. */
    private void probabilities(float[] logits, int[] idx, float[] out) {
        int m = out.length;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < m; i++) max = Math.max(max, logits[idx == null ? i : idx[i]]);
        float sum = 0f;
        for (int i = 0; i < m; i++) {
            float e = (float) Math.exp((logits[idx == null ? i : idx[i]] - max) / temperature);
            out[i] = e;
            sum += e;
        }
        for (int i = 0; i < m; i++) out[i] /= sum;
    }

    private int draw(float[] p, int[] idx, int count) {
        float total = 0f;
        for (int i = 0; i < count; i++) total += p[i];
        float r = rng.nextFloat() * total;
        float cum = 0f;
        for (int i = 0; i < count; i++) {
            cum += p[i];
            if (r < cum) return idx == null ? i : idx[i];
        }
        int last = count - 1;
        return idx == null ? last : idx[last];
    }

    public static int argmax(float[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) if (x[i] > x[best]) best = i;
        return best;
    }
}
