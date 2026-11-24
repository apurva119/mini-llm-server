package dev.apurva.llm.server.service;

import dev.apurva.llm.engine.generate.GenerationConfig;
import dev.apurva.llm.engine.generate.Generator;

public final class GenerationService {

    private final Generator generator;

    public GenerationService(Generator generator) {
        this.generator = generator;
    }

    public Generator.Result generate(String prompt, GenerationConfig config) {
        return generator.generate(prompt, config);
    }
}