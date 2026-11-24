package dev.apurva.llm.server.api;

import com.fasterxml.jackson.annotation.JsonProperty;

public record Choice(

        @JsonProperty("index")
        int index,

        @JsonProperty("message")
        Message message,

        @JsonProperty("finish_reason")
        String finishReason

) {
}
