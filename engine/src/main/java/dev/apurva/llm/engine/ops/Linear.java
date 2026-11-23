package dev.apurva.llm.engine.ops;

import dev.apurva.llm.engine.model.Tensor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.stream.IntStream;

/**
 * Matrix-vector product {@code y = W x} with {@code W} of shape {@code [out, in]} (row-major),
 * using the Vector API for SIMD and parallel row blocks for large matrices.
 */
public final class Linear {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    /** Parallelise when the matrix has at least this many elements. */
    private static final int PARALLEL_THRESHOLD = 1 << 16;
    private static final int ROWS_PER_TASK = 32;

    private final float[] weight;
    private final int outDim;
    private final int inDim;

    public Linear(float[] weight, int outDim, int inDim) {
        if ((long) outDim * inDim != weight.length) {
            throw new IllegalArgumentException("weight length " + weight.length + " != " + outDim + "x" + inDim);
        }
        this.weight = weight;
        this.outDim = outDim;
        this.inDim = inDim;
    }

    public Linear(Tensor t) {
        this(t.data(), t.rows(), t.cols());
        if (t.rank() != 2) throw new IllegalArgumentException("Linear needs a 2-D tensor: " + t);
    }

    public int outDim() { return outDim; }
    public int inDim() { return inDim; }

    /** Computes {@code y[0..out) = W x[0..in)}. */
    public void forward(float[] x, float[] y) {
        matvec(weight, outDim, inDim, x, y);
    }

    public static void matvec(float[] w, int rows, int cols, float[] x, float[] y) {
        if (x.length < cols) throw new IllegalArgumentException("x too short: " + x.length + " < " + cols);
        if (y.length < rows) throw new IllegalArgumentException("y too short: " + y.length + " < " + rows);
        if ((long) rows * cols >= PARALLEL_THRESHOLD && rows > ROWS_PER_TASK) {
            int tasks = (rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
            IntStream.range(0, tasks).parallel().forEach(t -> {
                int r0 = t * ROWS_PER_TASK;
                int r1 = Math.min(rows, r0 + ROWS_PER_TASK);
                for (int r = r0; r < r1; r++) y[r] = dot(w, r * cols, x, 0, cols);
            });
        } else {
            for (int r = 0; r < rows; r++) y[r] = dot(w, r * cols, x, 0, cols);
        }
    }

    /** Plain scalar reference implementation (used by tests). */
    public static void matvecScalar(float[] w, int rows, int cols, float[] x, float[] y) {
        for (int r = 0; r < rows; r++) {
            float s = 0f;
            for (int c = 0; c < cols; c++) s += w[r * cols + c] * x[c];
            y[r] = s;
        }
    }

    /** SIMD dot product of {@code a[aOff..aOff+n)} and {@code b[bOff..bOff+n)}. */
    public static float dot(float[] a, int aOff, float[] b, int bOff, int n) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;
        int bound = SPECIES.loopBound(n);
        for (; i < bound; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOff + i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            acc = va.fma(vb, acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) sum += a[aOff + i] * b[bOff + i];
        return sum;
    }
}
