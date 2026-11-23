package dev.apurva.llm.engine.runtime;


import dev.apurva.llm.engine.model.ModelConfig;

/**
 * Per-sequence mutable state: activation scratch buffers and the KV cache.
 * Weights are shared and immutable, so many states may run against one model concurrently.
 */
public final class InferenceState {
    public final float[] x;        // residual stream            [hidden]
    public final float[] xb;       // normed input / scratch     [hidden]
    public final float[] xb2;      // sub-layer output           [hidden]
    public final float[] q;        // queries                    [nHeads*headDim]
    public final float[] k;        // keys (this token)          [nKvHeads*headDim]
    public final float[] v;        // values (this token)        [nKvHeads*headDim]
    public final float[] ctx;      // attention context          [nHeads*headDim]
    public final float[] att;      // attention scores           [maxSeq]
    public final float[] hb;       // MLP gate / activated       [intermediate]
    public final float[] hb2;      // MLP up                     [intermediate]
    public final float[] logits;   //                            [vocab]
    public final float[][] keyCache;   // [layer][maxSeq*kvDim]
    public final float[][] valueCache; // [layer][maxSeq*kvDim]

    public InferenceState(ModelConfig c) {
        x = new float[c.hiddenSize()];
        xb = new float[c.hiddenSize()];
        xb2 = new float[c.hiddenSize()];
        q = new float[c.qDim()];
        k = new float[c.kvDim()];
        v = new float[c.kvDim()];
        ctx = new float[c.qDim()];
        att = new float[c.maxSeqLen()];
        hb = new float[c.intermediateSize()];
        hb2 = new float[c.intermediateSize()];
        logits = new float[c.vocabSize()];
        keyCache = new float[c.numLayers()][c.maxSeqLen() * c.kvDim()];
        valueCache = new float[c.numLayers()][c.maxSeqLen() * c.kvDim()];
    }
}
