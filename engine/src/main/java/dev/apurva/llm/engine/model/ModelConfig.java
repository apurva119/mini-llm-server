package dev.apurva.llm.engine.model;

import dev.apurva.llm.engine.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Architecture hyper-parameters read from a Hugging Face style {@code config.json}. */
public record ModelConfig(
        int hiddenSize,
        int intermediateSize,
        int vocabSize,
        int numHeads,
        int numKvHeads,
        int numLayers,
        int maxSeqLen,
        int headDim,
        float rmsNormEps,
        double ropeTheta,
        boolean tieWordEmbeddings,
        int bosTokenId,
        int eosTokenId) {

    public ModelConfig {
        if (hiddenSize <= 0 || intermediateSize <= 0 || vocabSize <= 0 || numHeads <= 0
                || numKvHeads <= 0 || numLayers <= 0 || maxSeqLen <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("All model dimensions must be positive");
        }
        if (numHeads % numKvHeads != 0) {
            throw new IllegalArgumentException("num_attention_heads must be a multiple of num_key_value_heads");
        }
        if (headDim % 2 != 0) {
            throw new IllegalArgumentException("head_dim must be even for RoPE");
        }
    }

    /** Size of the concatenated key (or value) vector per token. */
    public int kvDim() {
        return numKvHeads * headDim;
    }

    /** Size of the concatenated query vector per token. */
    public int qDim() {
        return numHeads * headDim;
    }

    public static ModelConfig load(Path configJson) throws IOException {
        return parse(Files.readString(configJson, StandardCharsets.UTF_8));
    }

    public static ModelConfig parse(String json) {
        Map<String, Object> m = Json.parseObject(json);

        Object act = m.get("hidden_act");
        if (act instanceof String a && !a.equals("silu")) {
            throw new UnsupportedOperationException("Unsupported hidden_act: " + a);
        }
        if (m.get("rope_scaling") != null) {
            throw new UnsupportedOperationException("rope_scaling is not supported");
        }

        int hidden = Json.getInt(m, "hidden_size", -1);
        int heads = Json.getInt(m, "num_attention_heads", -1);
        int kv = Json.getInt(m, "num_key_value_heads", heads);
        int headDim = Json.getInt(m, "head_dim", heads > 0 ? hidden / heads : -1);

        return new ModelConfig(
                hidden,
                Json.getInt(m, "intermediate_size", -1),
                Json.getInt(m, "vocab_size", -1),
                heads,
                kv,
                Json.getInt(m, "num_hidden_layers", -1),
                Json.getInt(m, "max_position_embeddings", -1),
                headDim,
                (float) Json.getDouble(m, "rms_norm_eps", 1e-5),
                Json.getDouble(m, "rope_theta", 10000.0),
                Json.getBool(m, "tie_word_embeddings", false),
                Json.getInt(m, "bos_token_id", 1),
                Json.getInt(m, "eos_token_id", 2));
    }
}
