package dev.apurva.llm.engine.ops;

/** Root-mean-square layer normalisation: {@code out = x / sqrt(mean(x^2) + eps) * weight}. */
public final class RMSNorm {
    private RMSNorm() {}

    /** {@code out} may alias {@code x}. Uses {@code weight.length} as the vector size. */
    public static void apply(float[] x, float[] weight, float eps, float[] out) {
        int n = weight.length;
        float ss = 0f;
        for (int i = 0; i < n; i++) ss += x[i] * x[i];
        float inv = (float) (1.0 / Math.sqrt(ss / n + eps));
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * weight[i];
    }
}
