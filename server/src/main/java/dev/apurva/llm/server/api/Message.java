package dev.apurva.llm.server.api;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * One chat message in an OpenAI-compatible request or response.
 */
public record Message(

        @JsonProperty("role")
        String role,

        @JsonProperty("content")
        String content

) {
}
