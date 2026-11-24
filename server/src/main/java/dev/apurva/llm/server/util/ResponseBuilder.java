package dev.apurva.llm.server.util;

import dev.apurva.llm.engine.generate.Generator;

import java.util.Map;
import java.util.List;
import java.util.LinkedHashMap;

public final class ResponseBuilder {

    private ResponseBuilder() {

    }


    public static Map<String, Object> resultJson(Generator.Result r, boolean includeText) {
        Map<String, Object> m = new LinkedHashMap<>();
        if (includeText) m.put("text", r.text());
        else m.put("done", true);
        m.put("finish_reason", r.finishReason());
        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", r.promptTokens());
        usage.put("generated_tokens", r.generatedTokens());
        usage.put("prefill_ms", Math.round(r.prefillMillis() * 10) / 10.0);
        usage.put("decode_ms", Math.round(r.decodeMillis() * 10) / 10.0);
        usage.put("tokens_per_second", Math.round(r.tokensPerSecond() * 10) / 10.0);
        m.put("usage", usage);
        return m;
    }

    public static Map<String, Object> chatJson(Generator.Result r, String modelName) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", r.text());

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", r.finishReason());

        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", r.promptTokens());
        usage.put("completion_tokens", r.generatedTokens());
        usage.put("total_tokens", r.promptTokens() + r.generatedTokens());

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", "chatcmpl-" + System.currentTimeMillis());
        response.put("object", "chat.completion");
        response.put("created", System.currentTimeMillis() / 1000);
        response.put("model", modelName);
        response.put("choices", List.of(choice));
        response.put("usage", usage);
        return response;
    }

    public static Map<String, Object> chunkJson(String id, long created, String modelName,
                                                 Map<String, Object> delta, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finishReason); // null serialises as JSON null

        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "chat.completion.chunk");
        chunk.put("created", created);
        chunk.put("model", modelName);
        chunk.put("choices", List.of(choice));
        return chunk;
    }
}





