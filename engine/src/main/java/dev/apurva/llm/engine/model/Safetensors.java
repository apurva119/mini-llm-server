package dev.apurva.llm.engine.model;

import dev.apurva.llm.engine.json.Json;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parser for the {@code .safetensors} container format:
 * <pre>
 *   [u64 little-endian header length N][N bytes JSON header][raw tensor data]
 * </pre>
 * This class only reads and validates metadata; no tensor data is read.
 */
public final class Safetensors {
    private static final long MAX_HEADER = 100L * 1024 * 1024;

    private final Map<String, TensorInfo> tensors;
    private final long dataStart;
    private final long fileSize;

    private Safetensors(Map<String, TensorInfo> tensors, long dataStart, long fileSize) {
        this.tensors = tensors;
        this.dataStart = dataStart;
        this.fileSize = fileSize;
    }

    public Map<String, TensorInfo> tensors() {
        return tensors;
    }

    public TensorInfo info(String name) {
        return tensors.get(name);
    }

    /** Absolute file offset at which the data section begins. */
    public long dataStart() {
        return dataStart;
    }

    public long fileSize() {
        return fileSize;
    }

    public static Safetensors read(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            return read(ch);
        }
    }

    public static Safetensors read(FileChannel ch) throws IOException {
        long size = ch.size();
        if (size < 8) throw new IOException("File too small to be safetensors");

        ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, lenBuf, 0);
        long headerLen = lenBuf.getLong(0);
        if (headerLen < 2 || headerLen > MAX_HEADER || 8 + headerLen > size) {
            throw new IOException("Invalid safetensors header length: " + headerLen);
        }

        ByteBuffer hb = ByteBuffer.allocate((int) headerLen);
        readFully(ch, hb, 8);
        String json = new String(hb.array(), StandardCharsets.UTF_8);

        long dataStart = 8 + headerLen;
        long dataLen = size - dataStart;

        Map<String, Object> header = Json.parseObject(json);
        Map<String, TensorInfo> out = new LinkedHashMap<>();

        for (Map.Entry<String, Object> e : header.entrySet()) {
            String name = e.getKey();
            if (name.equals("__metadata__")) continue;
            if (!(e.getValue() instanceof Map<?, ?> t)) {
                throw new IOException("Tensor entry is not an object: " + name);
            }
            DType dtype = DType.parse(String.valueOf(t.get("dtype")));
            if (!(t.get("shape") instanceof List<?> shapeList)) {
                throw new IOException("Missing shape for " + name);
            }
            int[] shape = new int[shapeList.size()];
            long numel = 1;
            for (int i = 0; i < shape.length; i++) {
                long d = ((Number) shapeList.get(i)).longValue();
                if (d < 0 || d > Integer.MAX_VALUE) throw new IOException("Bad dimension in " + name);
                shape[i] = (int) d;
                numel *= d;
            }
            if (!(t.get("data_offsets") instanceof List<?> off) || off.size() != 2) {
                throw new IOException("Missing data_offsets for " + name);
            }
            long begin = ((Number) off.get(0)).longValue();
            long end = ((Number) off.get(1)).longValue();
            if (begin < 0 || end < begin || end > dataLen) {
                throw new IOException("Offsets out of range for " + name);
            }
            if (end - begin != numel * dtype.bytes) {
                throw new IOException("Byte size does not match shape/dtype for " + name);
            }
            out.put(name, new TensorInfo(name, dtype, shape, begin, end));
        }
        return new Safetensors(out, dataStart, size);
    }

    private static void readFully(FileChannel ch, ByteBuffer buf, long pos) throws IOException {
        while (buf.hasRemaining()) {
            int n = ch.read(buf, pos + buf.position());
            if (n < 0) throw new IOException("Unexpected end of file");
        }
    }
}
