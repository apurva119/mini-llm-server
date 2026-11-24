package dev.apurva.llm.server;

import dev.apurva.llm.server.service.GenerationService;
import dev.apurva.llm.server.util.HttpUtil;
import dev.apurva.llm.server.util.ResponseBuilder;


import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import dev.apurva.llm.engine.generate.GenerationConfig;
import dev.apurva.llm.engine.generate.Generator;
import dev.apurva.llm.engine.json.Json;
import dev.apurva.llm.engine.layers.Transformer;
import dev.apurva.llm.engine.model.ModelConfig;
import dev.apurva.llm.engine.tokenizer.Tokenizer;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
/**
 * HTTP front-end for the inference engine.
 *
 * <pre>
 *   GET  /health
 *   GET  /v1/model
 *   GET  /v1/stats
 *   POST /v1/generate          {"prompt": "...", "max_tokens": 64, "temperature": 0.8,
 *                               "top_k": 40, "top_p": 0.95, "seed": 42, "stream": false}
 *   POST /v1/chat/completions  {"messages": [{"role": "user", "content": "..."}], "stream": false, ...}
 * </pre>
 *
 * Admission control: at most {@code maxConcurrent} requests generate at once. Up to {@code maxQueue}
 * more may wait (for at most {@code queueTimeoutMs}); anything beyond that is rejected with 429.
 * A running request is aborted after {@code requestTimeoutMs}, checked each time text is produced.
 */
public final class LlmServer {
    private static final int MAX_BODY_BYTES = 1 << 20;
    private static final int MAX_MESSAGES = 64;
    private static final Set<String> ROLES = Set.of("system", "user", "assistant");

    private final Generator generator;
    private final GenerationService generationService;
    private final ModelConfig config;
    private final Semaphore slots;
    private final int maxQueue;
    private final long queueTimeoutMs;
    private final long requestTimeoutMs;

    // Live gauges and lifetime counters, exposed at /v1/stats.
    private final AtomicInteger active = new AtomicInteger();
    private final AtomicInteger waiting = new AtomicInteger();
    private final AtomicLong completed = new AtomicLong();
    private final AtomicLong rejectedQueueFull = new AtomicLong();
    private final AtomicLong rejectedQueueTimeout = new AtomicLong();
    private final AtomicLong clientCancelled = new AtomicLong();
    private final AtomicLong timedOut = new AtomicLong();

    /** Thrown from the text callback to stop a generation early. */
    private static final class GenerationAborted extends RuntimeException {
        GenerationAborted() {
            super("request time limit exceeded", null, false, false);
        }
    }

    public LlmServer(Generator generator, int maxConcurrent) {
        this(generator, maxConcurrent, 8, 30_000, 60_000);
    }

    public LlmServer(Generator generator, int maxConcurrent, int maxQueue,
                     long queueTimeoutMs, long requestTimeoutMs) {
        if (maxConcurrent < 1) throw new IllegalArgumentException("maxConcurrent must be >= 1");
        if (maxQueue < 0) throw new IllegalArgumentException("maxQueue must be >= 0");
        this.generator = generator;
        this.generationService = new GenerationService(generator);
        this.config = generator.model().config();
        this.slots = new Semaphore(maxConcurrent);
        this.maxQueue = maxQueue;
        this.queueTimeoutMs = queueTimeoutMs;
        this.requestTimeoutMs = requestTimeoutMs;
    }

    public static void main(String[] args) throws Exception {
        Path modelDir = Path.of("models/tinyllama-15M");
        int port = 8080;
        int concurrent = 2;
        int queue = 8;
        long queueTimeoutMs = 30_000;
        long requestTimeoutMs = 60_000;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model" -> modelDir = Path.of(args[++i]);
                case "--port" -> port = Integer.parseInt(args[++i]);
                case "--concurrency" -> concurrent = Integer.parseInt(args[++i]);
                case "--queue" -> queue = Integer.parseInt(args[++i]);
                case "--queue-timeout-ms" -> queueTimeoutMs = Long.parseLong(args[++i]);
                case "--request-timeout-ms" -> requestTimeoutMs = Long.parseLong(args[++i]);
                case "--help", "-h" -> {
                    System.out.println("Usage: LlmServer [--model DIR] [--port N] [--concurrency N] [--queue N]"
                            + " [--queue-timeout-ms N] [--request-timeout-ms N]");
                    return;
                }
                default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }

        System.err.println("Loading model from " + modelDir.toAbsolutePath() + " ...");
        Transformer model = Transformer.load(modelDir);
        Tokenizer tokenizer = Tokenizer.load(modelDir);
        LlmServer app = new LlmServer(new Generator(model, tokenizer),
                concurrent, queue, queueTimeoutMs, requestTimeoutMs);

        app.start(new InetSocketAddress(port));
        System.err.println("Listening on http://localhost:" + port
                + " (concurrency=" + concurrent + ", queue=" + queue + ")");
    }

    /**
     * Starts the HTTP API on the supplied address.
     * Port 0 lets the operating system select an available port.
     */
    public HttpServer start(InetSocketAddress address) throws IOException {
        HttpServer http = HttpServer.create(address, 0);
        http.createContext("/health", safe(this::health));
        http.createContext("/v1/model", safe(this::modelInfo));
        http.createContext("/v1/stats", safe(this::stats));
        http.createContext("/v1/generate", safe(this::generate));
        http.createContext("/v1/chat/completions", safe(this::chatCompletions));
        http.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        http.start();
        return http;
    }
    // ---- handlers ------------------------------------------------------------

    private void health(HttpExchange ex) throws IOException {
        if (!HttpUtil.method(ex, "GET")) return;
        HttpUtil.send(ex, 200, Map.of("status", "ok"));
    }

    private void modelInfo(HttpExchange ex) throws IOException {
        if (!HttpUtil.method(ex, "GET")) return;
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("hidden_size", config.hiddenSize());
        m.put("intermediate_size", config.intermediateSize());
        m.put("num_layers", config.numLayers());
        m.put("num_heads", config.numHeads());
        m.put("num_kv_heads", config.numKvHeads());
        m.put("vocab_size", config.vocabSize());
        m.put("max_seq_len", config.maxSeqLen());
        HttpUtil.send(ex, 200, m);
    }

    private void stats(HttpExchange ex) throws IOException {
        if (!HttpUtil.method(ex, "GET")) return;
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("active", active.get());
        m.put("waiting", waiting.get());
        m.put("completed", completed.get());
        m.put("rejected_queue_full", rejectedQueueFull.get());
        m.put("rejected_queue_timeout", rejectedQueueTimeout.get());
        m.put("client_cancelled", clientCancelled.get());
        m.put("timed_out", timedOut.get());
        HttpUtil.send(ex, 200, m);
    }

    private void generate(HttpExchange ex) throws IOException {
        if (!HttpUtil.method(ex, "POST")) return;

        String prompt;
        GenerationConfig cfg;
        boolean stream;
        try {
            Map<String, Object> req = readJson(ex);
            prompt = Json.getString(req, "prompt", null);
            if (prompt == null) throw new IllegalArgumentException("\"prompt\" (string) is required");
            cfg = parseConfig(req);
            stream = Json.getBool(req, "stream", false);
        } catch (RuntimeException e) {
            badRequest(ex, e);
            return;
        }
        execute(ex, prompt, cfg, stream, null);
    }

    private void chatCompletions(HttpExchange ex) throws IOException {
        if (!HttpUtil.method(ex, "POST")) return;

        String prompt;
        GenerationConfig cfg;
        String modelName;
        boolean stream;
        try {
            Map<String, Object> req = readJson(ex);
            prompt = chatPrompt(req.get("messages"));
            cfg = parseConfig(req);
            stream = Json.getBool(req, "stream", false);
            modelName = Json.getString(req, "model", "tinyllama");
        } catch (RuntimeException e) {
            badRequest(ex, e);
            return;
        }
        execute(ex, prompt, cfg, stream, modelName);
    }

    // ---- shared generation path ----------------------------------------------

    /**
     * The one place that validates the prompt length, applies admission control, and runs the
     * generation. {@code chatModel} is null for /v1/generate and the model name for chat responses.
     */
    private void execute(HttpExchange ex, String prompt, GenerationConfig cfg,
                         boolean stream, String chatModel) throws IOException {
        // Reject oversized prompts before taking a slot or committing to a streaming response.
        if (generator.tokenizer().encode(prompt, true, false).length >= config.maxSeqLen()) {
            HttpUtil.send(ex, 400, Map.of("error", "Prompt does not fit in the context window of "
                    + config.maxSeqLen() + " tokens"));
            return;
        }
        if (!admit(ex)) return;
        try {
            long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(requestTimeoutMs);
            if (stream && chatModel != null) streamChatResponse(ex, prompt, cfg, deadline, chatModel);
            else if (stream) streamResponse(ex, prompt, cfg, deadline);
            else plainResponse(ex, prompt, cfg, deadline, chatModel);
        } finally {
            active.decrementAndGet();
            slots.release();
        }
    }

    /**
     * Takes a generation slot, or sends 429 (queue full) / 503 (waited too long) and returns false.
     * On true the caller must decrement {@code active} and release {@code slots}.
     */
    private boolean admit(HttpExchange ex) throws IOException {
        if (slots.tryAcquire()) {
            active.incrementAndGet();
            return true;
        }
        if (waiting.incrementAndGet() > maxQueue) {
            waiting.decrementAndGet();
            rejectedQueueFull.incrementAndGet();
            ex.getResponseHeaders().set("Retry-After", "1");
            HttpUtil.send(ex, 429, Map.of("error", "Server is busy: queue is full"));
            return false;
        }
        boolean got = false;
        try {
            got = slots.tryAcquire(queueTimeoutMs, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            waiting.decrementAndGet();
        }
        if (!got) {
            rejectedQueueTimeout.incrementAndGet();
            ex.getResponseHeaders().set("Retry-After", "1");
            HttpUtil.send(ex, 503, Map.of("error", "Server is busy: timed out waiting for a slot"));
            return false;
        }
        active.incrementAndGet();
        return true;
    }

    private void plainResponse(HttpExchange ex, String prompt, GenerationConfig cfg,
                               long deadline, String chatModel) throws IOException {
        Generator.Result r;
        try {
            r = generator.generate(prompt, cfg, piece -> checkDeadline(deadline));
        } catch (GenerationAborted e) {
            timedOut.incrementAndGet();
            HttpUtil.send(ex, 504, Map.of("error", "Generation exceeded the request time limit"));
            return;
        } catch (IllegalArgumentException e) {
            HttpUtil.send(ex, 400, Map.of("error", String.valueOf(e.getMessage())));
            return;
        }
        completed.incrementAndGet();
        HttpUtil.send(ex, 200, chatModel == null ? ResponseBuilder.resultJson(r, true) : ResponseBuilder.chatJson(r, chatModel));
    }

    private void streamResponse(HttpExchange ex, String prompt, GenerationConfig cfg,
                                long deadline) throws IOException {
        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        ex.sendResponseHeaders(200, 0); // chunked
        try (OutputStream out = ex.getResponseBody()) {
            try {
                Generator.Result r = generator.generate(prompt, cfg, piece -> {
                    checkDeadline(deadline);
                    try {
                        writeEvent(out, Map.of("text", piece));
                    } catch (IOException e) {
                        throw new UncheckedIOException(e); // client went away: abort generation
                    }
                });
                writeEvent(out, ResponseBuilder.resultJson(r, false));
                completed.incrementAndGet();
            } catch (GenerationAborted e) {
                timedOut.incrementAndGet();
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("done", true);
                m.put("finish_reason", "timeout");
                writeEvent(out, m);
            } catch (UncheckedIOException e) {
                clientCancelled.incrementAndGet();
            }
        } catch (IOException e) {
            clientCancelled.incrementAndGet(); // failed while writing the final event or closing
        }
    }

    /**
     * OpenAI-style streaming: a role chunk, one chunk per piece of text, a final chunk carrying
     * finish_reason, then the literal "data: [DONE]". A time-limit abort ends with finish_reason "timeout".
     */
    private void streamChatResponse(HttpExchange ex, String prompt, GenerationConfig cfg,
                                    long deadline, String modelName) throws IOException {
        String id = "chatcmpl-" + System.currentTimeMillis();
        long created = System.currentTimeMillis() / 1000;
        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        ex.sendResponseHeaders(200, 0); // chunked
        try (OutputStream out = ex.getResponseBody()) {
            try {
                Map<String, Object> roleDelta = new LinkedHashMap<>();
                roleDelta.put("role", "assistant");
                roleDelta.put("content", "");
                writeEvent(out, ResponseBuilder.chunkJson(id, created, modelName, roleDelta, null));

                Generator.Result r = generator.generate(prompt, cfg, piece -> {
                    checkDeadline(deadline);
                    try {
                        writeEvent(out, ResponseBuilder.chunkJson(id, created, modelName, Map.of("content", piece), null));
                    } catch (IOException e) {
                        throw new UncheckedIOException(e); // client went away: abort generation
                    }
                });
                writeEvent(out, ResponseBuilder.chunkJson(id, created, modelName, Map.of(), r.finishReason()));
                writeDone(out);
                completed.incrementAndGet();
            } catch (GenerationAborted e) {
                timedOut.incrementAndGet();
                writeEvent(out, ResponseBuilder.chunkJson(id, created, modelName, Map.of(), "timeout"));
                writeDone(out);
            } catch (UncheckedIOException e) {
                clientCancelled.incrementAndGet();
            }
        } catch (IOException e) {
            clientCancelled.incrementAndGet(); // failed while writing a chunk or closing
        }
    }



    private static void writeDone(OutputStream out) throws IOException {
        out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static void checkDeadline(long deadlineNanos) {
        if (System.nanoTime() - deadlineNanos > 0) throw new GenerationAborted();
    }

    // ---- request parsing -----------------------------------------------------

    private GenerationConfig parseConfig(Map<String, Object> req) {
        int maxTokens = Json.getInt(req, "max_tokens", 128);
        float temperature = (float) Json.getDouble(req, "temperature", 0.8);
        int topK = Json.getInt(req, "top_k", 40);
        float topP = (float) Json.getDouble(req, "top_p", 0.95);
        long seed = req.containsKey("seed") ? Json.getLong(req, "seed", 0) : System.nanoTime();

        if (maxTokens < 0 || maxTokens > config.maxSeqLen()) {
            throw new IllegalArgumentException("max_tokens must be between 0 and " + config.maxSeqLen());
        }
        if (!(temperature >= 0f)) throw new IllegalArgumentException("temperature must be >= 0");
        if (topK < 0) throw new IllegalArgumentException("top_k must be >= 0");
        if (!(topP > 0f && topP <= 1f)) throw new IllegalArgumentException("top_p must be in (0, 1]");
        return new GenerationConfig(maxTokens, temperature, topK, topP, seed);
    }

    /** Validates {@code messages} and flattens it into a plain-text prompt ending with an assistant cue. */
    private static String chatPrompt(Object messages) {
        if (!(messages instanceof List<?> list) || list.isEmpty()) {
            throw new IllegalArgumentException("\"messages\" must be a non-empty array");
        }
        if (list.size() > MAX_MESSAGES) {
            throw new IllegalArgumentException("\"messages\" may contain at most " + MAX_MESSAGES + " entries");
        }
        StringBuilder sb = new StringBuilder();
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> m)) {
                throw new IllegalArgumentException("each message must be an object");
            }
            Object role = m.get("role");
            Object content = m.get("content");
            if (!(role instanceof String r) || !ROLES.contains(r)) {
                throw new IllegalArgumentException("message \"role\" must be one of " + ROLES);
            }
            if (!(content instanceof String c) || c.isBlank()) {
                throw new IllegalArgumentException("message \"content\" must be a non-empty string");
            }
            sb.append(r).append(": ").append(c).append('\n');
        }
        sb.append("assistant: ");
        return sb.toString();
    }

    // ---- response building ---------------------------------------------------





    // ---- helpers -------------------------------------------------------------

    /** Turns any unexpected exception into a 500 instead of a dropped connection. */
    private static HttpHandler safe(HttpHandler h) {
        return ex -> {
            try {
                h.handle(ex);
            } catch (Throwable t) {
                System.err.println("Unhandled error on " + ex.getRequestURI() + ": " + t);
                try {
                    HttpUtil.send(ex, 500, Map.of("error", "internal server error"));
                } catch (Throwable ignored) {
                    // headers were already sent, or the client is gone
                }
            } finally {
                ex.close();
            }
        };
    }

    private static Map<String, Object> readJson(HttpExchange ex) throws IOException {
        byte[] body = readBody(ex);
        return Json.parseObject(new String(body, StandardCharsets.UTF_8));
    }

    private static void badRequest(HttpExchange ex, RuntimeException e) throws IOException {
        String msg = e.getMessage() != null ? e.getMessage() : "invalid request";
        HttpUtil.send(ex, 400, Map.of("error", msg));
    }

    private static void writeEvent(OutputStream out, Object json) throws IOException {
        out.write(("data: " + Json.stringify(json) + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }


    private static byte[] readBody(HttpExchange ex) throws IOException {
        try (var in = ex.getRequestBody()) {
            byte[] body = in.readNBytes(MAX_BODY_BYTES + 1);
            if (body.length > MAX_BODY_BYTES) throw new IllegalArgumentException("Request body too large");
            return body;
        }
    }


}
