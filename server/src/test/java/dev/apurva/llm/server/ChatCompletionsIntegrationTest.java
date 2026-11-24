package dev.apurva.llm.server;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import dev.apurva.llm.engine.generate.Generator;
import dev.apurva.llm.engine.layers.Transformer;
import dev.apurva.llm.engine.tokenizer.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assumptions;

import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class ChatCompletionsIntegrationTest {

    private static HttpServer server;
    private static HttpClient client;
    private static String baseUrl;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @BeforeAll
    static void startServer() throws Exception {
        Path modelDir = findModelDirectory();

        Assumptions.assumeTrue(
                modelDir != null,
                "Local tinyllama-15M model not available; integration tests skipped"
        );

        Transformer model = Transformer.load(modelDir);
        Tokenizer tokenizer = Tokenizer.load(modelDir);

        Generator generator = new Generator(model, tokenizer);

        LlmServer application = new LlmServer(generator, 1, 2, 3000, 30000);

        server = application.start(new InetSocketAddress("127.0.0.1", 0));

        baseUrl = "http://127.0.0.1:" + server.getAddress().getPort();

        client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    @AfterAll
    static void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    private static Path findModelDirectory() {
        Path direct = Path.of("models", "tinyllama-15M");
        if (Files.isRegularFile(direct.resolve("config.json"))) {
            return direct;
        }

        Path parent = Path.of("..", "models", "tinyllama-15M");
        if (Files.isRegularFile(parent.resolve("config.json"))) {
            return parent;
        }

        return null;
    }

    private HttpResponse<String> get(String path) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(
                URI.create(baseUrl + path)
        ).GET().build();

        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> post(String body) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(
                URI.create(baseUrl + "/v1/chat/completions")
        )
                .header("Content-Type", "application/json")
                .timeout(Duration.ofSeconds(30))
                .POST(HttpRequest.BodyPublishers.ofString(
                        body, StandardCharsets.UTF_8
                ))
                .build();

        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private String validChatRequest(boolean stream) {
        return """
                {
                  "model": "tinyllama-test",
                  "messages": [
                    {
                      "role": "user",
                      "content": "Say hello."
                    }
                  ],
                  "max_tokens": 2,
                  "temperature": 0.8,
                  "top_k": 1,
                  "top_p": 1.0,
                  "seed": 42,
                  "stream": %s
                }
                """.formatted(stream);
    }

    @Test
    void healthEndpointReturnsOk() throws Exception {
        HttpResponse<String> response = get("/health");

        assertEquals(200, response.statusCode());

        JsonNode json = MAPPER.readTree(response.body());

        assertEquals("ok", json.get("status").asText());
    }

    @Test
    void modelEndpointReturnsModelConfiguration() throws Exception {
        HttpResponse<String> response = get("/v1/model");

        assertEquals(200, response.statusCode());

        JsonNode json = MAPPER.readTree(response.body());

        assertTrue(json.has("hidden_size"));
        assertTrue(json.has("num_layers"));
        assertTrue(json.has("vocab_size"));
        assertTrue(json.has("max_seq_len"));
    }

    @Test
    void chatEndpointRejectsGetRequests() throws Exception {
        HttpResponse<String> response = get("/v1/chat/completions");

        assertEquals(405, response.statusCode());

        assertEquals(
                "POST",
                response.headers().firstValue("Allow").orElse("")
        );
    }

    @Test
    void chatEndpointRejectsEmptyMessages() throws Exception {
        HttpResponse<String> response = post(
                """
                {
                  "messages": []
                }
                """
        );

        assertEquals(400, response.statusCode());

        JsonNode json = MAPPER.readTree(response.body());

        assertTrue(json.has("error"));
    }

    @Test
    void chatEndpointReturnsOpenAiCompatibleResponse() throws Exception {
        HttpResponse<String> response = post(validChatRequest(false));

        assertEquals(200, response.statusCode());

        JsonNode json = MAPPER.readTree(response.body());

        assertEquals("chat.completion", json.get("object").asText());
        assertEquals("tinyllama-test", json.get("model").asText());

        JsonNode choices = json.get("choices");

        assertTrue(choices.isArray());
        assertFalse(choices.isEmpty());

        assertEquals(
                "assistant",
                choices.get(0).get("message").get("role").asText()
        );

        assertTrue(json.has("usage"));
        assertTrue(json.get("usage").has("total_tokens"));
    }

    @Test
    void chatEndpointReturnsStreamingChunksAndDoneMarker() throws Exception {
        HttpResponse<String> response = post(validChatRequest(true));

        assertEquals(200, response.statusCode());

        assertTrue(
                response.headers().firstValue("Content-Type")
                        .orElse("")
                        .contains("text/event-stream")
        );

        assertTrue(response.body().contains("chat.completion.chunk"));
        assertTrue(response.body().contains("[DONE]"));
    }

    @Test
    void rejectsUnknownMessageRole() throws Exception {
        HttpResponse<String> response = post("""
                {
                  "messages": [
                    {"role": "developer", "content": "Hello"}
                  ]
                }
                """);

        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("error"));
    }

    @Test
    void rejectsBlankMessageContent() throws Exception {
        HttpResponse<String> response = post("""
                {
                  "messages": [
                    {"role": "user", "content": "   "}
                  ]
                }
                """);

        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("error"));
    }

    @Test
    void rejectsMalformedJson() throws Exception {
        HttpResponse<String> response = post("{ invalid json");

        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("error"));
    }

    @Test
    void rejectsInvalidTopP() throws Exception {
        HttpResponse<String> response = post("""
                {
                  "messages": [
                    {"role": "user", "content": "Hello"}
                  ],
                  "top_p": 0
                }
                """);

        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("top_p"));
    }

    @Test
    void rejectsNegativeMaxTokens() throws Exception {
        HttpResponse<String> response = post("""
                {
                  "messages": [
                    {"role": "user", "content": "Hello"}
                  ],
                  "max_tokens": -1
                }
                """);

        assertEquals(400, response.statusCode());
        assertTrue(response.body().contains("max_tokens"));
    }

    @Test
    void statsEndpointExposesServerCounters() throws Exception {
        HttpResponse<String> response = get("/v1/stats");

        assertEquals(200, response.statusCode());

        JsonNode json = MAPPER.readTree(response.body());

        String[] fields = {
                "active",
                "waiting",
                "completed",
                "rejected_queue_full",
                "rejected_queue_timeout",
                "client_cancelled",
                "timed_out"
        };

        for (String field : fields) {
            assertTrue(json.has(field), "Missing stats field: " + field);
            assertTrue(json.get(field).isNumber());
            assertTrue(json.get(field).asLong() >= 0);
        }
    }

    @Test
    void handlesConcurrentChatRequestsWithinQueueCapacity() throws Exception {
        List<CompletableFuture<HttpResponse<String>>> futures = new ArrayList<>();

        for (int i = 0; i < 3; i++) {
            HttpRequest request = HttpRequest.newBuilder(
                    URI.create(baseUrl + "/v1/chat/completions")
            )
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(30))
                    .POST(HttpRequest.BodyPublishers.ofString(
                            validChatRequest(false),
                            StandardCharsets.UTF_8
                    ))
                    .build();

            futures.add(client.sendAsync(
                    request,
                    HttpResponse.BodyHandlers.ofString()
            ));
        }

        CompletableFuture.allOf(
                futures.toArray(CompletableFuture[]::new)
        ).get(45, TimeUnit.SECONDS);

        for (CompletableFuture<HttpResponse<String>> future : futures) {
            HttpResponse<String> response = future.join();

            assertEquals(200, response.statusCode(), response.body());

            JsonNode json = MAPPER.readTree(response.body());

            assertEquals("chat.completion", json.get("object").asText());
            assertEquals(1, json.get("choices").size());
        }
    }
}