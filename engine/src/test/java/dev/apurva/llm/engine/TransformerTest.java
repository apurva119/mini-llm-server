package dev.apurva.llm.engine;

import dev.apurva.llm.engine.generate.GenerationConfig;
import dev.apurva.llm.engine.generate.Generator;
import dev.apurva.llm.engine.model.ModelConfig;
import dev.apurva.llm.engine.layers.Transformer;
import dev.apurva.llm.engine.runtime.InferenceState;
import dev.apurva.llm.engine.tokenizer.Tokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class TransformerTest {
    @TempDir Path dir;
    Transformer model;
    Tokenizer tokenizer;

    @BeforeEach
    void setUp() throws Exception {
        TestModels.writeModelDir(dir, 123);
        model = Transformer.load(dir);
        tokenizer = Tokenizer.load(dir);
    }

    private float[] run(int... tokens) {
        InferenceState st = model.newState();
        float[] logits = null;
        for (int i = 0; i < tokens.length; i++) logits = model.forward(st, tokens[i], i, i == tokens.length - 1);
        return logits.clone();
    }

    @Test
    void configParsesAndLoads() {
        ModelConfig c = model.config();
        assertEquals(16, c.hiddenSize());
        assertEquals(2, c.numKvHeads());
        assertEquals(4, c.headDim());
        assertEquals(TestModels.VOCAB, c.vocabSize());
        assertTrue(c.tieWordEmbeddings());
    }

    @Test
    void logitsAreFiniteAndDeterministic() {
        float[] a = run(1, 260, 261, 262);
        float[] b = run(1, 260, 261, 262);
        assertEquals(TestModels.VOCAB, a.length);
        for (float f : a) assertTrue(Float.isFinite(f));
        assertArrayEquals(a, b, 0f);
    }

    @Test
    void historyAffectsOutput() {
        float[] a = run(1, 260, 261);
        float[] b = run(1, 262, 261); // same final token, different context
        double diff = 0;
        for (int i = 0; i < a.length; i++) diff += Math.abs(a[i] - b[i]);
        assertTrue(diff > 1e-3, "attention must make output depend on earlier tokens");
    }

    @Test
    void independentStatesDoNotInterfere() {
        InferenceState s1 = model.newState(), s2 = model.newState();
        model.forward(s1, 1, 0, false);
        model.forward(s2, 1, 0, false);
        model.forward(s2, 260, 1, false); // advance only s2
        float[] l1 = model.forward(s1, 260, 1, true).clone();
        float[] ref = run(1, 260);
        assertArrayEquals(ref, l1, 0f);
    }

    @Test
    void rejectsBadInputs() {
        InferenceState st = model.newState();
        assertThrows(IllegalArgumentException.class, () -> model.forward(st, -1, 0, true));
        assertThrows(IllegalArgumentException.class, () -> model.forward(st, TestModels.VOCAB, 0, true));
        assertThrows(IllegalArgumentException.class, () -> model.forward(st, 1, TestModels.MAX_SEQ, true));
    }

    @Test
    void generatorGreedyIsDeterministicAndRespectsLimits() {
        Generator g = new Generator(model, tokenizer);
        StringBuilder streamed = new StringBuilder();
        Generator.Result r1 = g.generate("abc", GenerationConfig.greedy(10), streamed::append);
        Generator.Result r2 = g.generate("abc", GenerationConfig.greedy(10));
        assertEquals(r1.text(), r2.text());
        assertEquals(r1.text(), streamed.toString());
        assertTrue(r1.generatedTokens() <= 10);
        assertEquals(3, r1.promptTokens()); // BOS, "▁ab", "c"
        assertTrue(r1.finishReason().equals("length") || r1.finishReason().equals("stop"));
    }

    @Test
    void generatorStopsAtContextWindowAndRejectsLongPrompts() {
        Generator g = new Generator(model, tokenizer);
        Generator.Result r = g.generate("abc", GenerationConfig.greedy(10_000));
        assertTrue(r.promptTokens() + r.generatedTokens() <= TestModels.MAX_SEQ);

        String huge = "a".repeat(TestModels.MAX_SEQ * 2);
        assertThrows(IllegalArgumentException.class, () -> g.generate(huge, GenerationConfig.greedy(1)));
    }

    @Test
    void sampledGenerationIsSeedReproducible() {
        Generator g = new Generator(model, tokenizer);
        GenerationConfig cfg = new GenerationConfig(20, 1.0f, 10, 0.9f, 99);
        assertEquals(g.generate("ab", cfg).text(), g.generate("ab", cfg).text());
    }
}
