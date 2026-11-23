package dev.apurva.llm.engine.model;

/** Element types that may appear in a safetensors file. */
public enum DType {
    F64(8), F32(4), F16(2), BF16(2), I64(8), I32(4), I16(2), I8(1), U8(1), BOOL(1);

    public final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

    public static DType parse(String s) {
        try {
            return DType.valueOf(s);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Unsupported safetensors dtype: " + s);
        }
    }
}
