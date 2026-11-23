package dev.apurva.llm.engine.ops;

/** SiLU / swish activation: {@code x * sigmoid(x)}. */
public final class SiLU {
    private SiLU() {}

    public static float apply(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    public static void inPlace(float[] x, int n) {
        for (int i = 0; i < n; i++) x[i] = apply(x[i]);
    }
}
