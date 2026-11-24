package dev.apurva.llm.server.util;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class HttpUtilTest {

    private HttpServer server;
    private HttpClient client;

    @BeforeEach
    void startServer() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);

        server.createContext("/echo", exchange -> {
            Map<String, Object> request = HttpUtil.readJson(exchange);
            HttpUtil.send(exchange, 200, request);
        });

        server.createContext("/post-only", exchange -> {
            if (HttpUtil.method(exchange, "POST")) {
                HttpUtil.send(exchange, 200, Map.of("ok", true));
            }
        });

        server.createContext("/body-limit", exchange -> {
            try {
                byte[] body = HttpUtil.readBody(exchange);
                HttpUtil.send(exchange, 200, Map.of("bytes", body.length));
            } catch (IllegalArgumentException e) {
                HttpUtil.send(exchange, 413, Map.of("error", e.getMessage()));
            }
        });

        server.start();
        client = HttpClient.newHttpClient();
    }

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    private URI uri(String path) {
        return URI.create(
                "http://127.0.0.1:" + server.getAddress().getPort() + path
        );
    }

    @Test
    void readJsonParsesRequestAndSendReturnsJson() throws Exception {
        String body = "{\"message\":\"hello\"}";

        HttpRequest request = HttpRequest.newBuilder(uri("/echo"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

        HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
        );

        assertEquals(200, response.statusCode());
        assertTrue(response.body().contains("\"message\":\"hello\""));
        assertTrue(response.headers()
                .firstValue("Content-Type").orElse("")
                .contains("application/json"));
    }

    @Test
    void methodRejectsUnexpectedHttpMethod() throws Exception {
        HttpRequest request = HttpRequest.newBuilder(uri("/post-only"))
                .GET()
                .build();

        HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
        );

        assertEquals(405, response.statusCode());
        assertEquals("POST", response.headers()
                .firstValue("Allow").orElse(""));
        assertTrue(response.body().contains("Method not allowed"));
    }

    @Test
    void readBodyRejectsRequestsLargerThanOneMiB() throws Exception {
        byte[] body = new byte[(1 << 20) + 1];

        HttpRequest request = HttpRequest.newBuilder(uri("/body-limit"))
                .POST(HttpRequest.BodyPublishers.ofByteArray(body))
                .build();

        HttpResponse<String> response = client.send(
                request,
                HttpResponse.BodyHandlers.ofString()
        );

        assertEquals(413, response.statusCode());
        assertTrue(response.body().contains("Request body too large"));
    }

    @Test
    void writeEventAndDoneUseCorrectSseFormat() throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();

        HttpUtil.writeEvent(output, Map.of("token", "hello"));
        HttpUtil.writeDone(output);

        String result = output.toString(StandardCharsets.UTF_8);

        assertEquals(
                "data: {\"token\":\"hello\"}\n\n" +
                "data: [DONE]\n\n",
                result
        );
    }
}