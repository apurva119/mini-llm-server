package dev.apurva.llm.engine.model;

import java.util.Arrays;

/**
 * Metadata for one tensor in a safetensors file.
 *
 * @param begin start offset relative to the start of the data section
 * @param end   end offset (exclusive) relative to the start of the data section
 */
public record TensorInfo(String name, DType dtype, int[] shape, long begin, long end) {

    public long numel() {
        long n = 1;
        for (int d : shape) n *= d;
        return n;
    }

    public long byteSize() {
        return end - begin;
    }

    @Override
    public String toString() {
        return name + " " + dtype + Arrays.toString(shape) + " [" + begin + "," + end + ")";
    }
}
