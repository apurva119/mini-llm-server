Mini LLM Server (Java 25)

A from-scratch LLM inference engine in pure Java: safetensors loading, a Llama-style transformer (RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU MLP), a SentencePiece tokenizer, sampling, a generation loop, a CLI and an HTTP server. The engine uses no ML frameworks and no native code. Python is used only for the reference scripts in tools/reference/. The engine targets CPU inference using the Java Vector API (an incubator module in JDK 25).

Status: work in progress. The engine is implemented and covered by unit tests on synthetic models. Verification against a PyTorch reference implementation on a real checkpoint is in progress (see Verification). Performance measurements will be added after correctness verification is complete.

Not a competitor to llama.cpp or GPU runtimes.

Features
Pure Java 25 transformer inference engine, no ML frameworks, no native code
Safetensors loader (F32 / F16 / BF16)
Llama architecture: RMSNorm, RoPE, grouped-query attention, SwiGLU
SentencePiece tokenizer with byte fallback
KV-cache based autoregressive generation
Temperature, top-k and top-p sampling
HTTP server with Server-Sent Events streaming
OpenAI-style /v1/chat/completions endpoint
Admission control: concurrency limit, bounded queue, timeouts
Command-line interface
Unit tests covering tokenizer, transformer, layers, sampling, safetensors loading, JSON parsing and math operators, on synthetic models

Current development focuses on CPU inference with float32 weights; quantization is not implemented yet.

Goals
Build a transformer inference engine from scratch in Java.
Verify correctness against a reference implementation.
Measure performance improvements one step at a time.
Keep the implementation readable and framework-free.
Layout
engine/   (no HTTP, no CLI code)
  model/      data only:  ModelConfig, DType, TensorInfo, Safetensors, Tensor, TensorLoader
  ops/        math primitives, model-independent:  Linear, RMSNorm, SiLU, Softmax, RoPE
  layers/     Llama-specific blocks built from ops:  Attention, MLP, DecoderLayer, Transformer
  runtime/    execution state:  InferenceState (activation buffers + KV cache)
  tokenizer/  Tokenizer, Utf8Stream
  sampling/   Sampler
  generate/   Generator, GenerationConfig
  json/       Json
  diag/       LaneCheck
server/   LlmServer  - HTTP API (JDK built-in com.sun.net.httpserver)
cli/      Main       - command-line generation
tools/reference/      - Python scripts that export PyTorch reference logits
models/   put model directories here (git-ignored)

Dependency direction:

server ─┐
        ├──► generate ──► layers ──┬──► ops
cli ────┘                          ├──► runtime
                                   └──► model
Requirements
JDK 25 (the build pins a Java 25 toolchain). The Vector API is an incubator module, so builds and runs pass --add-modules jdk.incubator.vector.
Gradle: use the wrapper. Run gradle wrapper once to generate it, then ./gradlew (Linux/macOS) or .\gradlew (Windows).
Memory: weights are loaded onto the Java heap as float[] (widened to float32), not memory-mapped, so heap size must exceed the model's float32 size.
Python 3.10+ only if you want to regenerate the reference fixtures.
Model directory

models/tinyllama-15M/ must contain (Hugging Face Llama format):

file	notes
config.json	hidden_size, num_attention_heads, num_key_value_heads, ...
model.safetensors	F32 / F16 / BF16 weights (widened to float32 on load)
tokenizer.model	SentencePiece model (Llama 2 tokenizer, 32000 vocab)
Build and run

Linux/macOS:

bash
./gradlew test

./gradlew :cli:run --args='--model models/tinyllama-15M --prompt "Once upon a time" --max-tokens 100'

./gradlew :server:run --args='--model models/tinyllama-15M --port 8080'

Windows (PowerShell):

powershell
.\gradlew test

.\gradlew :cli:run --args='--model models/tinyllama-15M --prompt "Once upon a time" --max-tokens 100'

.\gradlew :server:run --args='--model models/tinyllama-15M --port 8080'

Server flags: --concurrency N (simultaneous generations), --queue N (waiting requests), --queue-timeout-ms N, --request-timeout-ms N.

SIMD lane check: run dev.apurva.llm.engine.diag.LaneCheck with --add-modules jdk.incubator.vector.

HTTP API
endpoint	method	notes
/health	GET	liveness
/v1/model	GET	model dimensions
/v1/stats	GET	active / waiting requests and lifetime counters
/v1/generate	POST	raw prompt completion, optional SSE streaming
/v1/chat/completions	POST	OpenAI-style request and response, optional streaming
bash
curl -X POST localhost:8080/v1/generate -d '{
  "prompt": "Once upon a time", "max_tokens": 64,
  "temperature": 0.8, "top_k": 40, "top_p": 0.95, "seed": 42
}'

# streaming (Server-Sent Events); the last event has "done": true plus usage stats
curl -N -X POST localhost:8080/v1/generate -d '{"prompt": "Once upon a time", "stream": true}'

curl -X POST localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Once upon a time"}], "max_tokens": 64
}'

temperature: 0 is greedy. Chat streaming ("stream": true) sends OpenAI-style chat.completion.chunk events and ends with data: [DONE].

Error codes: 400 (bad request or prompt too long), 405 (wrong method), 429 (queue full, with Retry-After), 503 (timed out waiting for a slot), 504 (request time limit exceeded), 500 (unexpected error). Errors return {"error": "..."}.

The 15M checkpoint used for development exists to validate the engine and the API. It is a small story-generation model and is not instruction-tuned, so the chat endpoint speaks the protocol but replies read like story continuations rather than assistant answers.

Verification

Automated tests live in engine/src/test and run with ./gradlew test. They cover the tokenizer, transformer, layers, sampling, safetensors loading, the JSON parser and the math operators, using synthetic tiny models.

Reference comparison against PyTorch (in progress). ReferenceLogitsTest compares fp32 logits with logits exported by tools/reference/make_reference.py. It is skipped, not failed, when the model or fixtures are absent, so a skipped run is not evidence of correctness. To run it:

bash
python tools/reference/make_reference.py <hf-repo-id> engine/src/test/resources
./gradlew :engine:test --tests "*ReferenceLogitsTest"

This section will state the checkpoint, token ids, tolerance and measured error once the test passes and its fixtures are committed. Tokenizer parity with the reference tokenizer is not yet verified separately.

Deterministic mode. Start the JVM with -Dllm.scalar=true for plain single-threaded scalar math, used as the correctness baseline and for benchmarking.

Design notes
Weights are immutable and shared; all mutable data lives in runtime/InferenceState (activation buffers + KV cache), so concurrent requests are safe.
Linear uses the Vector API (FloatVector.SPECIES_PREFERRED, FMA) and splits large matrices into parallel row blocks. Weight shape is [out, in] row-major, matching safetensors, so no transposes.
RoPE uses the Hugging Face "rotate half" pairing (i, i + headDim/2), which matches HF checkpoints (including llama2.c models converted to HF). Raw llama2.c .bin files use adjacent pairs and are not supported.
Tied embeddings: if lm_head.weight is absent, the embedding matrix is reused as the output head.
Prefill feeds prompt tokens one at a time (no batching) and skips the LM head until the last one.
The tokenizer parses the SentencePiece protobuf directly and applies llama2.c-style score-based BPE merges with byte fallback. Streaming output buffers partial UTF-8 sequences.
Admission control: a semaphore limits concurrent generations, a bounded queue rejects overflow with 429, and a request time limit aborts long generations. The deadline and client-disconnect checks run when text is emitted, so a very long prefill is not interruptible.
Not implemented yet

Batched forward pass and continuous batching, int8 quantization, memory-mapped off-heap weights, rope_scaling variants, chat templates, a tokenizer.json loader, repetition penalty, and disconnect detection while a request is queued.

License

MIT. See LICENSE.