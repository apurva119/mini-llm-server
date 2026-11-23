package dev.apurva.llm.engine.model;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Set;

/** Memory-maps a safetensors file and materialises tensors by name as float32. */
public final class TensorLoader implements AutoCloseable {
    private final FileChannel channel;
    private final Safetensors index;

    public TensorLoader(Path file) throws IOException {
        this.channel = FileChannel.open(file, StandardOpenOption.READ);
        try {
            this.index = Safetensors.read(channel);
        } catch (IOException | RuntimeException e) {
            channel.close();
            throw e;
        }
    }

    public boolean has(String name) {
        return index.tensors().containsKey(name);
    }

    public Set<String> names() {
        return index.tensors().keySet();
    }

    public TensorInfo info(String name) {
        return index.info(name);
    }

    public Tensor get(String name) throws IOException {
        TensorInfo ti = index.info(name);
        if (ti == null) throw new IllegalArgumentException("No such tensor: " + name);

        int n = Math.toIntExact(ti.numel());
        float[] out = new float[n];
        if (n > 0) {
            MappedByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY,
                    index.dataStart() + ti.begin(), ti.byteSize());
            buf.order(ByteOrder.LITTLE_ENDIAN);
            switch (ti.dtype()) {
                case F32 -> buf.asFloatBuffer().get(out);
                case F16 -> { for (int i = 0; i < n; i++) out[i] = Float.float16ToFloat(buf.getShort(i * 2)); }
                case BF16 -> { for (int i = 0; i < n; i++) out[i] = Float.intBitsToFloat((buf.getShort(i * 2) & 0xFFFF) << 16); }
                case F64 -> { for (int i = 0; i < n; i++) out[i] = (float) buf.getDouble(i * 8); }
                default -> throw new UnsupportedOperationException(
                        "Cannot load " + ti.dtype() + " tensor as float: " + name);
            }
        }
        return new Tensor(name, ti.shape(), out);
    }

    @Override
    public void close() throws IOException {
        channel.close();
    }
}
