package dev.apurva.llm.server.api;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public record ChatRequest(

        @JsonProperty("model")
        String model,

        @JsonProperty("messages")
        List<Message> messages,

        @JsonProperty("temperature")
        Double temperature,

        @JsonProperty("max_tokens")
        Integer maxTokens

) {
}
