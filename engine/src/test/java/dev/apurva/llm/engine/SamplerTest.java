package dev.apurva.llm.engine;

import dev.apurva.llm.engine.sampling.Sampler;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SamplerTest {
    private final float[] logits = {0.1f, 3.0f, 1.0f, 2.5f, -1.0f};

    @Test
    void greedyPicksArgmax() {
        assertEquals(1, Sampler.greedy().sample(logits));
        assertEquals(1, new Sampler(0f, 3, 0.5f, 7).sample(logits));
    }

    @Test
    void sameSeedIsReproducible() {
        Sampler a = new Sampler(1.0f, 0, 1f, 42), b = new Sampler(1.0f, 0, 1f, 42);
        for (int i = 0; i < 50; i++) assertEquals(a.sample(logits), b.sample(logits));
    }

    @Test
    void topKRestrictsCandidates() {
        Sampler s = new Sampler(2.0f, 2, 1f, 1);
        for (int i = 0; i < 300; i++) {
            int t = s.sample(logits);
            assertTrue(t == 1 || t == 3, "got " + t);
        }
    }

    @Test
    void topPRestrictsCandidates() {
        // Probabilities at T=1 put >0.5 on token 1, so top_p=0.3 must always pick it.
        Sampler s = new Sampler(1.0f, 0, 0.3f, 1);
        for (int i = 0; i < 100; i++) assertEquals(1, s.sample(logits));
    }

    @Test
    void highTemperatureEventuallyVisitsManyTokens() {
        Sampler s = new Sampler(5.0f, 0, 1f, 3);
        boolean[] seen = new boolean[logits.length];
        for (int i = 0; i < 2000; i++) seen[s.sample(logits)] = true;
        for (boolean b : seen) assertTrue(b);
    }

    @Test
    void handlesNegativeLogitsInSortedPath() {
        float[] neg = {-5f, -1f, -3f};
        Sampler s = new Sampler(1.0f, 1, 1f, 1);
        assertEquals(1, s.sample(neg));
    }

    @Test
    void validatesArguments() {
        assertThrows(IllegalArgumentException.class, () -> new Sampler(-1f, 0, 1f, 0));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(1f, -1, 1f, 0));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(1f, 0, 0f, 0));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(1f, 0, 1.5f, 0));
    }
}
