package dev.apurva.llm.server.api;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChatRequestTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void deserializesChatRequestUsingOpenAiFieldNames() throws Exception {
        String json = """
                {
                  "model": "tinyllama",
                  "messages": [
                    {
                      "role": "user",
                      "content": "Hello"
                    }
                  ],
                  "temperature": 0.7,
                  "max_tokens": 32
                }
                """;

        ChatRequest request = mapper.readValue(json, ChatRequest.class);

        assertEquals("tinyllama", request.model());
        assertEquals(1, request.messages().size());
        assertEquals(new Message("user", "Hello"), request.messages().get(0));
        assertEquals(0.7, request.temperature(), 0.000001);
        assertEquals(32, request.maxTokens());
    }

    @Test
    void serializesMessageUsingRoleAndContentFields() throws Exception {
        Message message = new Message("assistant", "Hello there");

        JsonNode json = mapper.readTree(mapper.writeValueAsString(message));

        assertEquals("assistant", json.get("role").asText());
        assertEquals("Hello there", json.get("content").asText());
        assertEquals(2, json.size());
    }

    @Test
    void serializesMaxTokensUsingSnakeCaseFieldName() throws Exception {
        ChatRequest request = new ChatRequest(
                "tinyllama",
                java.util.List.of(new Message("user", "Hello")),
                0.7,
                32
        );

        JsonNode json = mapper.valueToTree(request);

        assertTrue(json.has("max_tokens"));
        assertFalse(json.has("maxTokens"));
    }
}