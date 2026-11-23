package dev.apurva.llm.engine;

import dev.apurva.llm.engine.tokenizer.Tokenizer;
import dev.apurva.llm.engine.tokenizer.Utf8Stream;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerTest {
    // ids: 0 unk, 1 bos, 2 eos, 3..258 bytes, 259 "▁", 260 a, 261 b, 262 c, 263 "▁a", 264 ab, 265 "▁ab", 266 abc
    private final Tokenizer tok = Tokenizer.fromSentencePiece(TestModels.tokenizerBytes());

    @Test
    void parsesVocabAndSpecialTokens() {
        assertEquals(TestModels.VOCAB, tok.vocabSize());
        assertEquals(1, tok.bosId());
        assertEquals(2, tok.eosId());
    }

    @Test
    void mergesByBestScore() {
        // "▁abc": ▁,a,b,c -> ab(-2) beats ▁a(-3) -> ▁,ab,c -> ▁ab(-1) -> ▁ab,c
        assertArrayEquals(new int[] {1, 265, 262}, tok.encode("abc", true, false));
        assertArrayEquals(new int[] {265, 262, 2}, tok.encode("abc", false, true));
        assertArrayEquals(new int[] {1}, tok.encode("", true, false));
    }

    @Test
    void usesByteFallbackForUnknownCharacters() {
        int[] ids = tok.encode("\u00e9", false, false); // é = C3 A9
        assertArrayEquals(new int[] {259, 3 + 0xC3, 3 + 0xA9}, ids);
        assertEquals("\u00e9", tok.decode(ids));
    }

    @Test
    void decodeRoundTripsAndStripsDummyPrefix() {
        assertEquals("abc", tok.decode(tok.encode("abc", true, true)));
        assertEquals("a b", tok.decode(tok.encode("a b", false, false)));
    }

    @Test
    void utf8StreamHoldsBackPartialCharacters() {
        byte[] euro = "\u20ac".getBytes(StandardCharsets.UTF_8); // 3 bytes
        Utf8Stream s = new Utf8Stream();
        assertEquals("", s.push(new byte[] {euro[0]}));
        assertEquals("", s.push(new byte[] {euro[1]}));
        assertEquals("\u20ac", s.push(new byte[] {euro[2]}));
        assertEquals("hi", s.push("hi".getBytes(StandardCharsets.UTF_8)));
        assertEquals("", s.push(new byte[] {euro[0]}));
        assertEquals("\uFFFD", s.flush()); // dangling lead byte is replaced, not lost silently
    }
}
