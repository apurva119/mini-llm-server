package dev.apurva.llm.engine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.apurva.llm.engine.layers.Transformer;
import dev.apurva.llm.engine.runtime.InferenceState;
import dev.apurva.llm.engine.tokenizer.Tokenizer;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ReferenceComparisonTest {

    @Test
    void compareAgainstReference() throws Exception {

        Path modelDir = Path.of("../models/tinyllama-15M").normalize();

        Transformer model = Transformer.load(modelDir);
        Tokenizer tokenizer = Tokenizer.load(modelDir);

        // Ensure tokenizer loads correctly
        assertTrue(tokenizer.vocabSize() > 0);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(
    Path.of("../python/reference.json").normalize().toFile()
);

        JsonNode ids = root.get("input_ids");
        JsonNode refLogits = root.get("logits");

        InferenceState state = model.newState();

        float[] logits = null;

        for (int i = 0; i < ids.size(); i++) {
            logits = model.forward(
                    state,
                    ids.get(i).asInt(),
                    i,
                    i == ids.size() - 1
            );
        }

        double maxErr = 0.0;
        double mse = 0.0;

        for (int i = 0; i < logits.length; i++) {
            double ref = refLogits.get(i).asDouble();
            double err = Math.abs(logits[i] - ref);

            if (err > maxErr)
                maxErr = err;

            mse += err * err;
        }

        mse /= logits.length;

        System.out.println();
        System.out.println("======================================");
        System.out.printf("Maximum Error : %.10f%n", maxErr);
        System.out.printf("Mean Sq Error : %.12e%n", mse);
        System.out.println("======================================");

        assertTrue(maxErr < 1e-4);
    }
}