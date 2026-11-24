package dev.apurva.llm.server.api;

import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;

/**
 * Handles the OpenAI-compatible /v1/chat/completions endpoint.
 */
public final class ChatHandler {

    public ChatHandler() {
    }

    public void handle(HttpExchange exchange) throws IOException {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}