package dev.apurva.llm.engine.ops;

/** Numerically stable softmax. */
public final class Softmax {
    private Softmax() {}

    public static void inPlace(float[] x, int off, int len) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) max = Math.max(max, x[off + i]);
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            float e = (float) Math.exp(x[off + i] - max);
            x[off + i] = e;
            sum += e;
        }
        float inv = 1f / sum;
        for (int i = 0; i < len; i++) x[off + i] *= inv;
    }
}
