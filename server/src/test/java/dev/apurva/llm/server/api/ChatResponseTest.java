package dev.apurva.llm.server.api;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ChatResponseTest {

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void serializesCompleteChatResponseUsingOpenAiFieldNames() throws Exception {
        ChatResponse response = new ChatResponse(
                "chatcmpl-test",
                "chat.completion",
                1234567890L,
                "tinyllama",
                List.of(new Choice(
                        0,
                        new Message("assistant", "Hello there"),
                        "stop"
                )),
                new Usage(10, 5, 15)
        );

        JsonNode json = mapper.valueToTree(response);

        assertEquals("chatcmpl-test", json.get("id").asText());
        assertEquals("chat.completion", json.get("object").asText());
        assertEquals(1234567890L, json.get("created").asLong());
        assertEquals("tinyllama", json.get("model").asText());

        JsonNode choice = json.get("choices").get(0);
        assertEquals(0, choice.get("index").asInt());
        assertEquals("assistant", choice.get("message").get("role").asText());
        assertEquals("Hello there", choice.get("message").get("content").asText());
        assertEquals("stop", choice.get("finish_reason").asText());

        JsonNode usage = json.get("usage");
        assertEquals(10, usage.get("prompt_tokens").asInt());
        assertEquals(5, usage.get("completion_tokens").asInt());
        assertEquals(15, usage.get("total_tokens").asInt());
    }

    @Test
    void deserializesCompleteChatResponse() throws Exception {
        String json = """
                {
                  "id": "chatcmpl-test",
                  "object": "chat.completion",
                  "created": 1234567890,
                  "model": "tinyllama",
                  "choices": [
                    {
                      "index": 0,
                      "message": {
                        "role": "assistant",
                        "content": "Hello there"
                      },
                      "finish_reason": "stop"
                    }
                  ],
                  "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                  }
                }
                """;

        ChatResponse response = mapper.readValue(json, ChatResponse.class);

        assertEquals("chatcmpl-test", response.id());
        assertEquals(1, response.choices().size());
        assertEquals("Hello there", response.choices().get(0).message().content());
        assertEquals("stop", response.choices().get(0).finishReason());
        assertEquals(15, response.usage().totalTokens());
    }
}