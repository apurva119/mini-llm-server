package dev.apurva.llm.engine;

import dev.apurva.llm.engine.model.DType;
import dev.apurva.llm.engine.model.Safetensors;
import dev.apurva.llm.engine.model.Tensor;
import dev.apurva.llm.engine.model.TensorLoader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class SafetensorsTest {
    @TempDir Path dir;

    @Test
    void loadsF32F16AndBF16() throws Exception {
        Map<String, byte[]> raw = new LinkedHashMap<>();
        Map<String, String> dt = new LinkedHashMap<>();
        Map<String, int[]> sh = new LinkedHashMap<>();

        float[] vals = {1.5f, -2f, 0.25f, 8f, 3f, -0.5f};
        ByteBuffer f32 = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer f16 = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer bf16 = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : vals) {
            f32.putFloat(v);
            f16.putShort(Float.floatToFloat16(v));
            bf16.putShort((short) (Float.floatToIntBits(v) >>> 16));
        }
        raw.put("a", f32.array());  dt.put("a", "F32");  sh.put("a", new int[] {2, 3});
        raw.put("b", f16.array());  dt.put("b", "F16");  sh.put("b", new int[] {3, 2});
        raw.put("c", bf16.array()); dt.put("c", "BF16"); sh.put("c", new int[] {6});

        Path f = dir.resolve("t.safetensors");
        TestModels.writeRaw(f, raw, dt, sh);

        Safetensors st = Safetensors.read(f);
        assertEquals(3, st.tensors().size());
        assertEquals(DType.F16, st.info("b").dtype());

        try (TensorLoader loader = new TensorLoader(f)) {
            assertTrue(loader.has("a"));
            assertFalse(loader.has("zzz"));
            for (String n : new String[] {"a", "b", "c"}) {
                Tensor t = loader.get(n);
                assertArrayEquals(vals, t.data(), 0f); // all values exactly representable
            }
            Tensor a = loader.get("a");
            assertEquals(2, a.rows());
            assertEquals(3, a.cols());
            assertEquals(8f, a.get(1, 0), 0f);
            assertThrows(IllegalArgumentException.class, () -> loader.get("missing"));
        }
    }

    @Test
    void rejectsCorruptFiles() throws IOException {
        Path tiny = dir.resolve("tiny");
        Files.write(tiny, new byte[] {1, 2, 3});
        assertThrows(IOException.class, () -> Safetensors.read(tiny));

        // header claims 1000 bytes but file has only 20
        ByteBuffer bb = ByteBuffer.allocate(20).order(ByteOrder.LITTLE_ENDIAN);
        bb.putLong(1000);
        Path bad = dir.resolve("bad");
        Files.write(bad, bb.array());
        assertThrows(IOException.class, () -> Safetensors.read(bad));

        // offsets that do not match shape * dtype size
        String header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        ByteBuffer b2 = ByteBuffer.allocate(8 + header.length() + 8).order(ByteOrder.LITTLE_ENDIAN);
        b2.putLong(header.length());
        b2.put(header.getBytes());
        Path bad2 = dir.resolve("bad2");
        Files.write(bad2, b2.array());
        assertThrows(IOException.class, () -> Safetensors.read(bad2));
    }
}
