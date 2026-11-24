package dev.apurva.llm.engine.bench;

import dev.apurva.llm.engine.ops.Linear;
import org.openjdk.jmh.annotations.*;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(1)
@State(Scope.Thread)
public class LinearBenchmark {

    @Param({"512", "1024", "2048"})
    public int size;

    private float[] w;
    private float[] x;
    private float[] y;

    @Setup
    public void setup() {
        Random rnd = new Random(42);

        w = new float[size * size];
        x = new float[size];
        y = new float[size];

        for (int i = 0; i < w.length; i++) {
            w[i] = rnd.nextFloat() - 0.5f;
        }

        for (int i = 0; i < x.length; i++) {
            x[i] = rnd.nextFloat() - 0.5f;
        }
    }

    @Benchmark
    public void simd() {
        Linear.matvec(w, size, size, x, y);
    }

    @Benchmark
    public void scalar() {
        Linear.matvecScalar(w, size, size, x, y);
    }
}