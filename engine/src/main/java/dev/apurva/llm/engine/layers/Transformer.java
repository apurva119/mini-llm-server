package dev.apurva.llm.engine.layers;

import dev.apurva.llm.engine.model.ModelConfig;
import dev.apurva.llm.engine.model.Tensor;
import dev.apurva.llm.engine.model.TensorLoader;

import dev.apurva.llm.engine.runtime.InferenceState;
import dev.apurva.llm.engine.ops.Linear;
import dev.apurva.llm.engine.ops.RMSNorm;
import dev.apurva.llm.engine.ops.RoPE;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A complete Llama-style decoder-only transformer. Immutable after loading and safe to share between
 * threads; all mutable data lives in {@link InferenceState}.
 */
public final class Transformer {
    private final ModelConfig config;
    private final float[] embedding;      // [vocab * hidden]
    private final DecoderLayer[] layers;
    private final float[] finalNorm;
    private final Linear lmHead;

    private Transformer(ModelConfig config, float[] embedding, DecoderLayer[] layers,
                        float[] finalNorm, Linear lmHead) {
        this.config = config;
        this.embedding = embedding;
        this.layers = layers;
        this.finalNorm = finalNorm;
        this.lmHead = lmHead;
    }

    public ModelConfig config() {
        return config;
    }

    public InferenceState newState() {
        return new InferenceState(config);
    }

    /** Loads {@code config.json} and {@code model.safetensors} from a model directory. */
    public static Transformer load(Path modelDir) throws IOException {
        Path cfgPath = modelDir.resolve("config.json");
        Path weights = modelDir.resolve("model.safetensors");
        if (!Files.exists(cfgPath)) throw new IOException("Missing " + cfgPath);
        if (!Files.exists(weights)) throw new IOException("Missing " + weights);
        ModelConfig cfg = ModelConfig.load(cfgPath);
        try (TensorLoader loader = new TensorLoader(weights)) {
            return load(cfg, loader);
        }
    }

    public static Transformer load(ModelConfig c, TensorLoader loader) throws IOException {
        Tensor embed = expect(loader.get("model.embed_tokens.weight"), c.vocabSize(), c.hiddenSize());
        RoPE rope = new RoPE(c.headDim(), c.maxSeqLen(), c.ropeTheta());

        DecoderLayer[] layers = new DecoderLayer[c.numLayers()];
        for (int i = 0; i < layers.length; i++) {
            String p = "model.layers." + i + ".";
            Linear wq = new Linear(expect(loader.get(p + "self_attn.q_proj.weight"), c.qDim(), c.hiddenSize()));
            Linear wk = new Linear(expect(loader.get(p + "self_attn.k_proj.weight"), c.kvDim(), c.hiddenSize()));
            Linear wv = new Linear(expect(loader.get(p + "self_attn.v_proj.weight"), c.kvDim(), c.hiddenSize()));
            Linear wo = new Linear(expect(loader.get(p + "self_attn.o_proj.weight"), c.hiddenSize(), c.qDim()));
            Attention attn = new Attention(wq, wk, wv, wo, c.numHeads(), c.numKvHeads(), c.headDim(), rope);

            Linear gate = new Linear(expect(loader.get(p + "mlp.gate_proj.weight"), c.intermediateSize(), c.hiddenSize()));
            Linear up = new Linear(expect(loader.get(p + "mlp.up_proj.weight"), c.intermediateSize(), c.hiddenSize()));
            Linear down = new Linear(expect(loader.get(p + "mlp.down_proj.weight"), c.hiddenSize(), c.intermediateSize()));

            float[] n1 = expectVec(loader.get(p + "input_layernorm.weight"), c.hiddenSize());
            float[] n2 = expectVec(loader.get(p + "post_attention_layernorm.weight"), c.hiddenSize());
            layers[i] = new DecoderLayer(n1, n2, c.rmsNormEps(), attn, new MLP(gate, up, down));
        }

        float[] finalNorm = expectVec(loader.get("model.norm.weight"), c.hiddenSize());

        // Tied embeddings: reuse the embedding matrix as the output projection.
        Linear head;
        if (loader.has("lm_head.weight")) {
            head = new Linear(expect(loader.get("lm_head.weight"), c.vocabSize(), c.hiddenSize()));
        } else {
            head = new Linear(embed.data(), c.vocabSize(), c.hiddenSize());
        }
        return new Transformer(c, embed.data(), layers, finalNorm, head);
    }

    /**
     * Runs one token through the model at sequence position {@code pos}, updating the KV cache.
     *
     * @param wantLogits when false the final norm and LM head are skipped (used while prefilling a prompt)
     * @return the logits array held by {@code st} (reused on the next call), or null if not requested
     */
    public float[] forward(InferenceState st, int token, int pos, boolean wantLogits) {
        if (token < 0 || token >= config.vocabSize()) throw new IllegalArgumentException("token out of range: " + token);
        if (pos < 0 || pos >= config.maxSeqLen()) throw new IllegalArgumentException("position out of range: " + pos);

        int h = config.hiddenSize();
        System.arraycopy(embedding, token * h, st.x, 0, h);
        for (int l = 0; l < layers.length; l++) layers[l].forward(st, l, pos);

        if (!wantLogits) return null;
        RMSNorm.apply(st.x, finalNorm, config.rmsNormEps(), st.xb);
        lmHead.forward(st.xb, st.logits);
        return st.logits;
    }

    private static Tensor expect(Tensor t, int rows, int cols) {
        if (t.rank() != 2 || t.rows() != rows || t.cols() != cols) {
            throw new IllegalArgumentException(t.name() + " has shape " + java.util.Arrays.toString(t.shape())
                    + ", expected [" + rows + ", " + cols + "]");
        }
        return t;
    }

    private static float[] expectVec(Tensor t, int n) {
        if (t.rank() != 1 || t.numel() != n) {
            throw new IllegalArgumentException(t.name() + " has shape " + java.util.Arrays.toString(t.shape())
                    + ", expected [" + n + "]");
        }
        return t.data();
    }
}
