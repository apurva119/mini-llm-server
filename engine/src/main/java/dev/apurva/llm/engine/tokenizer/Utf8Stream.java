package dev.apurva.llm.engine.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Turns a stream of raw bytes into text, holding back incomplete multi-byte UTF-8 sequences so
 * byte-fallback tokens never produce garbled characters when streaming.
 */
public final class Utf8Stream {
    private final ByteArrayOutputStream pending = new ByteArrayOutputStream();

    /** Appends bytes and returns whatever complete text is now available (possibly empty). */
    public String push(byte[] bytes) {
        pending.writeBytes(bytes);
        byte[] buf = pending.toByteArray();
        int complete = completePrefix(buf);
        if (complete == 0) return "";
        pending.reset();
        pending.write(buf, complete, buf.length - complete);
        return new String(buf, 0, complete, StandardCharsets.UTF_8);
    }

    /** Returns any remaining bytes (malformed ones become U+FFFD). */
    public String flush() {
        String s = pending.toString(StandardCharsets.UTF_8);
        pending.reset();
        return s;
    }

    private static int completePrefix(byte[] buf) {
        int len = buf.length;
        for (int k = 1; k <= Math.min(3, len); k++) {
            int b = buf[len - k] & 0xFF;
            if ((b & 0xC0) == 0x80) continue; // continuation byte, keep looking for the lead byte
            int need = (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 1;
            return need > k ? len - k : len;
        }
        return len;
    }
}
