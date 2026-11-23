package dev.apurva.llm.cli;

import dev.apurva.llm.engine.generate.GenerationConfig;
import dev.apurva.llm.engine.generate.Generator;
import dev.apurva.llm.engine.layers.Transformer;
import dev.apurva.llm.engine.tokenizer.Tokenizer;

import java.nio.file.Path;

/** Command-line text generation: {@code Main --model DIR --prompt "Once upon a time"}. */
public final class Main {
    public static void main(String[] args) throws Exception {
        Path modelDir = Path.of("models/tinyllama-15M");
        String prompt = "Once upon a time";
        int maxTokens = 128;
        float temperature = 0.8f;
        int topK = 40;
        float topP = 0.95f;
        long seed = System.nanoTime();

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model" -> modelDir = Path.of(args[++i]);
                case "--prompt" -> prompt = args[++i];
                case "--max-tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--temperature" -> temperature = Float.parseFloat(args[++i]);
                case "--top-k" -> topK = Integer.parseInt(args[++i]);
                case "--top-p" -> topP = Float.parseFloat(args[++i]);
                case "--seed" -> seed = Long.parseLong(args[++i]);
                case "--help", "-h" -> {
                    System.out.println("""
                            Usage: Main [options]
                              --model DIR         model directory (config.json, model.safetensors, tokenizer.model)
                              --prompt TEXT       prompt text
                              --max-tokens N      max new tokens (default 128)
                              --temperature F     0 = greedy (default 0.8)
                              --top-k N           0 disables (default 40)
                              --top-p F           1 disables (default 0.95)
                              --seed N            RNG seed""");
                    return;
                }
                default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }

        long t0 = System.nanoTime();
        Transformer model = Transformer.load(modelDir);
        Tokenizer tokenizer = Tokenizer.load(modelDir);
        System.err.printf("Loaded model in %.0f ms%n", (System.nanoTime() - t0) / 1e6);

        Generator gen = new Generator(model, tokenizer);
        GenerationConfig cfg = new GenerationConfig(maxTokens, temperature, topK, topP, seed);

        System.out.print(prompt);
        System.out.flush();
        Generator.Result r = gen.generate(prompt, cfg, piece -> {
            System.out.print(piece);
            System.out.flush();
        });
        System.out.println();
        System.err.printf("[%d prompt tokens, %d generated, %s, prefill %.0f ms, %.1f tok/s]%n",
                r.promptTokens(), r.generatedTokens(), r.finishReason(), r.prefillMillis(), r.tokensPerSecond());
    }
}
