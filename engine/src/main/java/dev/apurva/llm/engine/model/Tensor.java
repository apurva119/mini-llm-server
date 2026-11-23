package dev.apurva.llm.engine.model;

import java.util.Arrays;

/**
 * Read-only model weights: a dense row-major float32 tensor.
 *
 * <p><b>Contract:</b> once constructed, the backing array returned by {@link #data()} must never be
 * modified. It is shared, without copying, by every {@code Linear} and by all concurrent inference
 * states. (Making this enforceable without copies would need a read-only {@code FloatBuffer} or
 * {@code MemorySegment}.)
 *
 * <p>Dense row-major float32 tensor. Weights of any on-disk dtype are widened to float32 at load time.
 * For a 2-D weight of shape {@code [out, in]}, {@code data[r * in + c]} is element (r, c).
 */
public final class Tensor {
    private final String name;
    private final int[] shape;
    private final float[] data;

    public Tensor(String name, int[] shape, float[] data) {
        long n = 1;
        for (int d : shape) n *= d;
        if (n != data.length) {
            throw new IllegalArgumentException("Shape " + Arrays.toString(shape) + " does not match data length " + data.length);
        }
        this.name = name;
        this.shape = shape.clone();
        this.data = data;
    }

    public String name() { return name; }
    public int[] shape() { return shape.clone(); }
    public int rank() { return shape.length; }
    public float[] data() { return data; }
    public int numel() { return data.length; }

    /** Size of the first dimension. */
    public int rows() { return shape[0]; }

    /** Product of all dimensions after the first (1 for a vector). */
    public int cols() {
        int c = 1;
        for (int i = 1; i < shape.length; i++) c *= shape[i];
        return c;
    }

    public float get(int i) { return data[i]; }
    public float get(int r, int c) { return data[r * cols() + c]; }

    @Override
    public String toString() {
        return "Tensor(" + name + ", " + Arrays.toString(shape) + ")";
    }
}
