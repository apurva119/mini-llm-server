package dev.apurva.llm.engine.generate;

import dev.apurva.llm.engine.layers.Transformer;
import dev.apurva.llm.engine.runtime.InferenceState;
import dev.apurva.llm.engine.sampling.Sampler;
import dev.apurva.llm.engine.tokenizer.Tokenizer;
import dev.apurva.llm.engine.tokenizer.Utf8Stream;

import java.util.function.Consumer;

/** The autoregressive loop: encode, prefill, then sample / append / repeat. Thread-safe. */
public final class Generator {

    public record Result(String text, int promptTokens, int generatedTokens, String finishReason,
                         double prefillMillis, double decodeMillis) {
        public double tokensPerSecond() {
            return decodeMillis > 0 ? generatedTokens / (decodeMillis / 1000.0) : 0.0;
        }
    }

    private final Transformer model;
    private final Tokenizer tokenizer;

    public Generator(Transformer model, Tokenizer tokenizer) {
        if (tokenizer.vocabSize() != model.config().vocabSize()) {
            throw new IllegalArgumentException("Tokenizer vocab (" + tokenizer.vocabSize()
                    + ") does not match model vocab (" + model.config().vocabSize() + ")");
        }
        this.model = model;
        this.tokenizer = tokenizer;
    }

    public Tokenizer tokenizer() { return tokenizer; }
    public Transformer model() { return model; }

    public Result generate(String prompt, GenerationConfig cfg) {
        return generate(prompt, cfg, null);
    }

    /**
     * @param onText receives each newly completed piece of text as it is produced (may be null)
     * @throws IllegalArgumentException if the prompt does not fit in the model's context window
     */
    public Result generate(String prompt, GenerationConfig cfg, Consumer<String> onText) {
        int maxSeq = model.config().maxSeqLen();
        int[] ids = tokenizer.encode(prompt, true, false);
        if (ids.length >= maxSeq) {
            throw new IllegalArgumentException("Prompt is " + ids.length + " tokens; context window is " + maxSeq);
        }
        int maxNew = Math.min(cfg.maxNewTokens(), maxSeq - ids.length);

        Sampler sampler = new Sampler(cfg.temperature(), cfg.topK(), cfg.topP(), cfg.seed());
        InferenceState st = model.newState();
        Utf8Stream utf8 = new Utf8Stream();
        StringBuilder text = new StringBuilder();

        long t0 = System.nanoTime();
        float[] logits = null;
        for (int i = 0; i < ids.length; i++) {
            logits = model.forward(st, ids[i], i, i == ids.length - 1);
        }
        long t1 = System.nanoTime();

        int pos = ids.length;
        int gen = 0;
        String reason = "length";
        while (gen < maxNew) {
            int next = sampler.sample(logits);
            if (next == tokenizer.eosId()) {
                reason = "stop";
                break;
            }
            gen++;
            emit(utf8.push(tokenizer.decodeToken(next)), text, onText);
            if (gen == maxNew) break;
            logits = model.forward(st, next, pos++, true);
        }
        emit(utf8.flush(), text, onText);
        long t2 = System.nanoTime();

        return new Result(text.toString(), ids.length, gen, reason, (t1 - t0) / 1e6, (t2 - t1) / 1e6);
    }

    private static void emit(String s, StringBuilder text, Consumer<String> onText) {
        if (s.isEmpty()) return;
        text.append(s);
        if (onText != null) onText.accept(s);
    }
}
