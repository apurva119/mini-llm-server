package dev.apurva.llm.engine.layers;

import dev.apurva.llm.engine.ops.RMSNorm;
import dev.apurva.llm.engine.runtime.InferenceState;

/** One pre-norm transformer block: {@code x += attn(norm(x)); x += mlp(norm(x))}. */
public final class DecoderLayer {
    private final float[] attnNorm;
    private final float[] ffnNorm;
    private final float eps;
    private final Attention attention;
    private final MLP mlp;

    public DecoderLayer(float[] attnNorm, float[] ffnNorm, float eps, Attention attention, MLP mlp) {
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.eps = eps;
        this.attention = attention;
        this.mlp = mlp;
    }

    public void forward(InferenceState st, int layer, int pos) {
        RMSNorm.apply(st.x, attnNorm, eps, st.xb);
        attention.forward(st, layer, pos, st.xb, st.xb2);
        add(st.x, st.xb2);

        RMSNorm.apply(st.x, ffnNorm, eps, st.xb);
        mlp.forward(st.xb, st.hb, st.hb2, st.xb2);
        add(st.x, st.xb2);
    }

    private static void add(float[] dst, float[] src) {
        for (int i = 0; i < dst.length; i++) dst[i] += src[i];
    }
}
