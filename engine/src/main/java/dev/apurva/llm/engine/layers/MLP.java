package dev.apurva.llm.engine.layers;

import dev.apurva.llm.engine.ops.Linear;
import dev.apurva.llm.engine.ops.SiLU;

/** Gated feed-forward block: {@code down( silu(gate(x)) * up(x) )}. */
public final class MLP {
    private final Linear gate;
    private final Linear up;
    private final Linear down;

    public MLP(Linear gate, Linear up, Linear down) {
        this.gate = gate;
        this.up = up;
        this.down = down;
    }

    /**
     * @param hb  scratch of size intermediate (overwritten)
     * @param hb2 scratch of size intermediate (overwritten)
     * @param out receives the hidden-size result
     */
    public void forward(float[] x, float[] hb, float[] hb2, float[] out) {
        gate.forward(x, hb);
        up.forward(x, hb2);
        int n = gate.outDim();
        for (int i = 0; i < n; i++) hb[i] = SiLU.apply(hb[i]) * hb2[i];
        down.forward(hb, out);
    }
}
