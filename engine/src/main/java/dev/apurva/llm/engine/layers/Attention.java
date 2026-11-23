package dev.apurva.llm.engine.layers;

import dev.apurva.llm.engine.ops.Linear;
import dev.apurva.llm.engine.ops.RoPE;
import dev.apurva.llm.engine.ops.Softmax;
import dev.apurva.llm.engine.runtime.InferenceState;

/** Causal multi-head self-attention with grouped-query support and a KV cache. */
public final class Attention {
    private final Linear wq, wk, wv, wo;
    private final int nHeads, nKvHeads, headDim;
    private final RoPE rope;

    public Attention(Linear wq, Linear wk, Linear wv, Linear wo,
                     int nHeads, int nKvHeads, int headDim, RoPE rope) {
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = headDim;
        this.rope = rope;
    }

    /**
     * Processes the token at {@code pos}: projects Q/K/V, applies RoPE, appends K/V to the cache,
     * attends over positions {@code 0..pos}, and writes the output projection to {@code out}.
     */
    public void forward(InferenceState st, int layer, int pos, float[] x, float[] out) {
        int kvDim = nKvHeads * headDim;

        wq.forward(x, st.q);
        wk.forward(x, st.k);
        wv.forward(x, st.v);
        rope.apply(st.q, nHeads, pos);
        rope.apply(st.k, nKvHeads, pos);

        float[] kc = st.keyCache[layer];
        float[] vc = st.valueCache[layer];
        System.arraycopy(st.k, 0, kc, pos * kvDim, kvDim);
        System.arraycopy(st.v, 0, vc, pos * kvDim, kvDim);

        int group = nHeads / nKvHeads;
        float scale = (float) (1.0 / Math.sqrt(headDim));
        float[] att = st.att;
        float[] ctx = st.ctx;

        for (int h = 0; h < nHeads; h++) {
            int qOff = h * headDim;
            int kvOff = (h / group) * headDim;

            for (int t = 0; t <= pos; t++) {
                att[t] = Linear.dot(st.q, qOff, kc, t * kvDim + kvOff, headDim) * scale;
            }
            Softmax.inPlace(att, 0, pos + 1);

            int cOff = h * headDim;
            for (int d = 0; d < headDim; d++) ctx[cOff + d] = 0f;
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                int vOff = t * kvDim + kvOff;
                for (int d = 0; d < headDim; d++) ctx[cOff + d] += a * vc[vOff + d];
            }
        }
        wo.forward(ctx, out);
    }
}
