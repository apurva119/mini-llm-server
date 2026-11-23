package dev.apurva.llm.engine.diag;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Prints Vector API availability and SIMD width. Run with {@code --add-modules jdk.incubator.vector}. */
public final class LaneCheck {
    public static void main(String[] args) {
        VectorSpecies<Float> sp = FloatVector.SPECIES_PREFERRED;
        System.out.println("Java            : " + System.getProperty("java.version"));
        System.out.println("Vector API      : available");
        System.out.println("Float species   : " + sp);
        System.out.println("Float lanes     : " + sp.length() + " (" + sp.vectorBitSize() + "-bit)");
        System.out.println("CPU cores       : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Max heap (MB)   : " + Runtime.getRuntime().maxMemory() / (1024 * 1024));
    }
}
