package dev.apurva.llm.engine;

import dev.apurva.llm.engine.ops.Linear;
import dev.apurva.llm.engine.ops.RMSNorm;
import dev.apurva.llm.engine.ops.RoPE;
import dev.apurva.llm.engine.ops.SiLU;
import dev.apurva.llm.engine.ops.Softmax;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class OpsTest {
    private static float[] rand(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float) r.nextGaussian();
        return a;
    }

    @Test
    void linearMatchesScalarReference() {
        Random r = new Random(1);
        for (int[] shape : new int[][] {{3, 5}, {17, 33}, {300, 257}}) { // incl. parallel path + SIMD tails
            int rows = shape[0], cols = shape[1];
            float[] w = rand(r, rows * cols), x = rand(r, cols);
            float[] y = new float[rows], ref = new float[rows];
            new Linear(w, rows, cols).forward(x, y);
            Linear.matvecScalar(w, rows, cols, x, ref);
            assertArrayEquals(ref, y, 1e-3f);
        }
    }

    @Test
    void linearKnownValues() {
        float[] w = {1, 2, 3, 4, 5, 6};
        float[] y = new float[2];
        new Linear(w, 2, 3).forward(new float[] {1, 0, -1}, y);
        assertArrayEquals(new float[] {-2, -2}, y, 1e-6f);
    }

    @Test
    void rmsNormKnownValues() {
        float[] x = {3, 4};
        float[] out = new float[2];
        RMSNorm.apply(x, new float[] {1, 2}, 0f, out);
        float rms = (float) Math.sqrt((9 + 16) / 2.0);
        assertArrayEquals(new float[] {3 / rms, 8 / rms}, out, 1e-6f);
    }

    @Test
    void siluKnownValues() {
        assertEquals(0f, SiLU.apply(0f), 1e-7f);
        assertEquals(0.7310586f, SiLU.apply(1f), 1e-6f);
        assertEquals(-0.2689414f, SiLU.apply(-1f), 1e-6f);
        assertEquals(10f, SiLU.apply(10f), 1e-3f);
    }

    @Test
    void softmaxSumsToOneAndIsStable() {
        float[] x = {1000f, 1001f, 1002f, -5f};
        Softmax.inPlace(x, 0, 3);
        assertEquals(1.0, x[0] + x[1] + x[2], 1e-6);
        assertTrue(x[2] > x[1] && x[1] > x[0]);
        assertEquals(-5f, x[3], 0f); // outside range untouched
    }

    @Test
    void ropeIsIdentityAtZeroAndPreservesNorm() {
        RoPE rope = new RoPE(8, 32, 10000.0);
        Random r = new Random(2);
        float[] v = rand(r, 16); // 2 heads
        float[] copy = v.clone();
        rope.apply(copy, 2, 0);
        assertArrayEquals(v, copy, 1e-7f);

        rope.apply(copy, 2, 17);
        for (int h = 0; h < 2; h++) {
            double n0 = 0, n1 = 0;
            for (int i = 0; i < 8; i++) {
                n0 += v[h * 8 + i] * v[h * 8 + i];
                n1 += copy[h * 8 + i] * copy[h * 8 + i];
            }
            assertEquals(n0, n1, 1e-4);
        }
        assertFalse(java.util.Arrays.equals(v, copy));
    }

    @Test
    void ropeRelativePositionProperty() {
        // <rot(q, m), rot(k, n)> depends only on m - n.
        RoPE rope = new RoPE(8, 64, 10000.0);
        Random r = new Random(3);
        float[] q = rand(r, 8), k = rand(r, 8);
        float d1 = dotAt(rope, q, k, 10, 4);
        float d2 = dotAt(rope, q, k, 30, 24);
        assertEquals(d1, d2, 1e-4f);
    }

    private static float dotAt(RoPE rope, float[] q, float[] k, int m, int n) {
        float[] a = q.clone(), b = k.clone();
        rope.apply(a, 1, m);
        rope.apply(b, 1, n);
        return Linear.dot(a, 0, b, 0, 8);
    }
}
