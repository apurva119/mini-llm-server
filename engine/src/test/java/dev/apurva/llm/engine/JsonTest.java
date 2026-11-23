package dev.apurva.llm.engine;

import dev.apurva.llm.engine.json.Json;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonTest {
    @Test
    void parsesNestedStructures() {
        Map<String, Object> m = Json.parseObject(
                "{\"a\": 1, \"b\": [1.5, \"x\\n\\u0041\", true, null], \"c\": {\"d\": -2e2}}");
        assertEquals(1L, m.get("a"));
        List<?> b = (List<?>) m.get("b");
        assertEquals(1.5, (Double) b.get(0), 1e-12);
        assertEquals("x\nA", b.get(1));
        assertEquals(Boolean.TRUE, b.get(2));
        assertNull(b.get(3));
        assertEquals(-200.0, ((Number) ((Map<?, ?>) m.get("c")).get("d")).doubleValue(), 1e-12);
    }

    @Test
    void rejectsMalformedInput() {
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{\"a\": }"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("[1, 2"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{} extra"));
    }

    @Test
    void stringifyRoundTrips() {
        String s = Json.stringify(Map.of("k", List.of(1, 2.5, "q\"uote\n", true)));
        Map<String, Object> back = Json.parseObject(s);
        List<?> l = (List<?>) back.get("k");
        assertEquals("q\"uote\n", l.get(2));
        assertEquals(1L, l.get(0));
    }
}
