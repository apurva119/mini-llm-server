package dev.apurva.llm.engine.generate;

/** Per-request generation settings. */
public record GenerationConfig(int maxNewTokens, float temperature, int topK, float topP, long seed) {

    public GenerationConfig {
        if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must be >= 0");
    }

    public static GenerationConfig greedy(int maxNewTokens) {
        return new GenerationConfig(maxNewTokens, 0f, 0, 1f, 0L);
    }

    public static GenerationConfig defaults() {
        return new GenerationConfig(128, 0.8f, 40, 0.95f, System.nanoTime());
    }
}
