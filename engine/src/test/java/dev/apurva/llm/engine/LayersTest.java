package dev.apurva.llm.engine;

import dev.apurva.llm.engine.layers.Attention;
import dev.apurva.llm.engine.layers.MLP;
import dev.apurva.llm.engine.model.ModelConfig;
import dev.apurva.llm.engine.ops.Linear;
import dev.apurva.llm.engine.ops.RoPE;
import dev.apurva.llm.engine.runtime.InferenceState;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class LayersTest {
    private static float[] rand(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) r.nextGaussian();
        return a;
    }

    @Test
    void mlpMatchesFormula() {
        Random r = new Random(4);
        int h = 6, i = 10;
        float[] g = rand(r, i * h), u = rand(r, i * h), d = rand(r, h * i), x = rand(r, h);
        float[] out = new float[h];
        new MLP(new Linear(g, i, h), new Linear(u, i, h), new Linear(d, h, i))
                .forward(x, new float[i], new float[i], out);

        float[] gx = new float[i], ux = new float[i], act = new float[i], ref = new float[h];
        Linear.matvecScalar(g, i, h, x, gx);
        Linear.matvecScalar(u, i, h, x, ux);
        for (int k = 0; k < i; k++) act[k] = gx[k] / (1f + (float) Math.exp(-gx[k])) * ux[k];
        Linear.matvecScalar(d, h, i, act, ref);
        assertArrayEquals(ref, out, 1e-4f);
    }

    @Test
    void attentionAtPositionZeroReturnsProjectedValues() {
        // With a single visible position softmax is 1, so ctx head h == value of kv-head h/group.
        ModelConfig c = ModelConfig.parse(TestModels.CONFIG);
        Random r = new Random(5);
        int hid = c.hiddenSize(), qd = c.qDim(), kd = c.kvDim();
        float[] wq = rand(r, qd * hid), wk = rand(r, kd * hid), wv = rand(r, kd * hid), wo = rand(r, hid * qd);
        Attention att = new Attention(new Linear(wq, qd, hid), new Linear(wk, kd, hid), new Linear(wv, kd, hid),
                new Linear(wo, hid, qd), c.numHeads(), c.numKvHeads(), c.headDim(),
                new RoPE(c.headDim(), c.maxSeqLen(), c.ropeTheta()));

        float[] x = rand(r, hid), out = new float[hid];
        InferenceState st = new InferenceState(c);
        att.forward(st, 0, 0, x, out);

        float[] v = new float[kd], ctx = new float[qd], ref = new float[hid];
        Linear.matvecScalar(wv, kd, hid, x, v);
        int group = c.numHeads() / c.numKvHeads();
        for (int h = 0; h < c.numHeads(); h++)
            for (int d = 0; d < c.headDim(); d++)
                ctx[h * c.headDim() + d] = v[(h / group) * c.headDim() + d];
        Linear.matvecScalar(wo, hid, qd, ctx, ref);
        assertArrayEquals(ref, out, 1e-4f);
    }
}
