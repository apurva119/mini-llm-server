package dev.apurva.llm.engine.ops;

/**
 * Rotary positional embedding using the Hugging Face "rotate half" layout: within each head,
 * element {@code i} is paired with element {@code i + headDim/2}.
 */
public final class RoPE {
    private final int headDim;
    private final int half;
    private final int maxSeq;
    private final float[] cos;
    private final float[] sin;

    public RoPE(int headDim, int maxSeq, double theta) {
        if (headDim % 2 != 0) throw new IllegalArgumentException("headDim must be even");
        this.headDim = headDim;
        this.half = headDim / 2;
        this.maxSeq = maxSeq;
        this.cos = new float[maxSeq * half];
        this.sin = new float[maxSeq * half];
        for (int pos = 0; pos < maxSeq; pos++) {
            for (int i = 0; i < half; i++) {
                double invFreq = Math.pow(theta, -2.0 * i / headDim);
                double angle = pos * invFreq;
                cos[pos * half + i] = (float) Math.cos(angle);
                sin[pos * half + i] = (float) Math.sin(angle);
            }
        }
    }

    /** Rotates {@code nHeads} consecutive heads of {@code vec} in place for position {@code pos}. */
    public void apply(float[] vec, int nHeads, int pos) {
        if (pos < 0 || pos >= maxSeq) throw new IllegalArgumentException("position out of range: " + pos);
        int base = pos * half;
        for (int h = 0; h < nHeads; h++) {
            int o = h * headDim;
            for (int i = 0; i < half; i++) {
                float c = cos[base + i];
                float s = sin[base + i];
                float x0 = vec[o + i];
                float x1 = vec[o + i + half];
                vec[o + i] = x0 * c - x1 * s;
                vec[o + i + half] = x1 * c + x0 * s;
            }
        }
    }
}
