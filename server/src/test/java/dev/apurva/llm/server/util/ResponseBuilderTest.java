package dev.apurva.llm.server.util;

import dev.apurva.llm.engine.generate.Generator;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ResponseBuilderTest {

    private Generator.Result sampleResult() {
        return new Generator.Result(
                "Hello there",
                10,
                5,
                "stop",
                12.34,
                4.56
        );
    }

    @Test
    void resultJsonIncludesTextAndUsage() {
        Map<String, Object> result =
                ResponseBuilder.resultJson(sampleResult(), true);

        assertEquals("Hello there", result.get("text"));
        assertEquals("stop", result.get("finish_reason"));
        assertFalse(result.containsKey("done"));

        Map<?, ?> usage = (Map<?, ?>) result.get("usage");

        assertEquals(10, usage.get("prompt_tokens"));
        assertEquals(5, usage.get("generated_tokens"));
        assertEquals(12.3, usage.get("prefill_ms"));
        assertEquals(4.6, usage.get("decode_ms"));
        assertEquals(1096.5, usage.get("tokens_per_second"));
    }

    @Test
    void resultJsonCanOmitText() {
        Map<String, Object> result =
                ResponseBuilder.resultJson(sampleResult(), false);

        assertEquals(true, result.get("done"));
        assertFalse(result.containsKey("text"));
        assertEquals("stop", result.get("finish_reason"));
    }

    @Test
    void chatJsonBuildsOpenAiCompatibleResponse() {
        Map<String, Object> result =
                ResponseBuilder.chatJson(sampleResult(), "tinyllama");

        assertTrue(result.get("id").toString().startsWith("chatcmpl-"));
        assertEquals("chat.completion", result.get("object"));
        assertTrue((long) result.get("created") > 0);
        assertEquals("tinyllama", result.get("model"));

        List<?> choices = (List<?>) result.get("choices");
        assertEquals(1, choices.size());

        Map<?, ?> choice = (Map<?, ?>) choices.get(0);
        assertEquals(0, choice.get("index"));
        assertEquals("stop", choice.get("finish_reason"));

        Map<?, ?> message = (Map<?, ?>) choice.get("message");
        assertEquals("assistant", message.get("role"));
        assertEquals("Hello there", message.get("content"));

        Map<?, ?> usage = (Map<?, ?>) result.get("usage");
        assertEquals(10, usage.get("prompt_tokens"));
        assertEquals(5, usage.get("completion_tokens"));
        assertEquals(15, usage.get("total_tokens"));
    }

    @Test
    void chunkJsonBuildsStreamingResponse() {
        Map<String, Object> delta = Map.of("content", "Hello");

        Map<String, Object> result = ResponseBuilder.chunkJson(
                "chatcmpl-test",
                1234567890L,
                "tinyllama",
                delta,
                null
        );

        assertEquals("chatcmpl-test", result.get("id"));
        assertEquals("chat.completion.chunk", result.get("object"));
        assertEquals(1234567890L, result.get("created"));
        assertEquals("tinyllama", result.get("model"));

        List<?> choices = (List<?>) result.get("choices");
        assertEquals(1, choices.size());

        Map<?, ?> choice = (Map<?, ?>) choices.get(0);
        assertEquals(0, choice.get("index"));
        assertEquals(delta, choice.get("delta"));
        assertNull(choice.get("finish_reason"));
    }
}