package dev.apurva.llm.server.util;

import com.sun.net.httpserver.HttpExchange;
import dev.apurva.llm.engine.json.Json;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public final class HttpUtil {

    private static final int MAX_BODY_BYTES = 1 << 20;

    private HttpUtil() {
    }

    public static boolean method(HttpExchange ex, String expected) throws IOException {
        if (ex.getRequestMethod().equalsIgnoreCase(expected)) {
            return true;
        }

        ex.getResponseHeaders().set("Allow", expected);
        send(ex, 405, Map.of("error", "Method not allowed; use " + expected));
        return false;
    }

    public static Map<String, Object> readJson(HttpExchange ex) throws IOException {
        byte[] body = readBody(ex);
        return Json.parseObject(new String(body, StandardCharsets.UTF_8));
    }

    public static byte[] readBody(HttpExchange ex) throws IOException {
        try (var in = ex.getRequestBody()) {
            byte[] body = in.readNBytes(MAX_BODY_BYTES + 1);
            if (body.length > MAX_BODY_BYTES) {
                throw new IllegalArgumentException("Request body too large");
            }
            return body;
        }
    }

    public static void send(HttpExchange ex, int status, Object json) throws IOException {
        byte[] bytes = Json.stringify(json).getBytes(StandardCharsets.UTF_8);

        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);

        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    public static void writeEvent(OutputStream out, Object json) throws IOException {
        out.write(("data: " + Json.stringify(json) + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    public static void writeDone(OutputStream out) throws IOException {
        out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
        out.flush();
    }
}