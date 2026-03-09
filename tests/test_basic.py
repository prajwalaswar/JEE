"""
test_basic.py — Basic smoke tests for Multimodal Math Mentor.

Run: pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Math Tools ────────────────────────────────────────────────────────────────

class TestMathTools:
    def test_solve_quadratic(self):
        from backend.tools.math_tools import solve_equation
        result = solve_equation("x**2 - 5*x + 6 = 0", variable="x")
        assert result["success"] is True
        solutions = set(result["solutions"])
        assert "2" in solutions and "3" in solutions

    def test_solve_linear(self):
        from backend.tools.math_tools import solve_equation
        result = solve_equation("2*x + 4 = 10", variable="x")
        assert result["success"] is True
        assert "3" in result["solutions"]

    def test_differentiate_polynomial(self):
        from backend.tools.math_tools import differentiate
        result = differentiate("x**3 + 2*x**2 + x", variable="x")
        assert result["success"] is True
        # d/dx (x³+2x²+x) = 3x²+4x+1
        assert "3*x**2" in result["derivative"] or "3x" in result["derivative"]

    def test_integrate_polynomial(self):
        from backend.tools.math_tools import integrate_expression
        result = integrate_expression("x**2", variable="x")
        assert result["success"] is True
        # ∫x² dx = x³/3
        assert "x**3" in result["result"]

    def test_compute_limit(self):
        from backend.tools.math_tools import compute_limit
        result = compute_limit("sin(x)/x", variable="x", point="0")
        assert result["success"] is True
        assert result["limit"] in ("1", "1.00000000000000")

    def test_simplify(self):
        from backend.tools.math_tools import simplify_expression
        result = simplify_expression("(x**2 - 1)/(x - 1)")
        assert result["success"] is True
        assert "x + 1" in result["simplified"] or "x+1" in result["simplified"]

    def test_factor(self):
        from backend.tools.math_tools import factor_expression
        result = factor_expression("x**2 - 5*x + 6")
        assert result["success"] is True
        factored = result["factored"]
        assert "(x - 2)" in factored or "(x - 3)" in factored

    def test_evaluate(self):
        from backend.tools.math_tools import evaluate_expression
        result = evaluate_expression("x**2 + 1", substitutions={"x": 3.0})
        assert result["success"] is True
        assert abs(result["value"] - 10.0) < 1e-9

    def test_tool_dispatcher(self):
        from backend.tools.math_tools import run_math_tool
        result = run_math_tool("solve_equation", equation_str="x - 7 = 0", variable="x")
        assert result["success"] is True
        assert "7" in result["solutions"]

    def test_unknown_tool(self):
        from backend.tools.math_tools import run_math_tool
        result = run_math_tool("nonexistent_tool")
        assert result["success"] is False


# ── Models ────────────────────────────────────────────────────────────────────

class TestModels:
    def test_parsed_problem_defaults(self):
        from backend.models import ParsedProblem, MathTopic
        p = ParsedProblem(problem_text="Test problem")
        assert p.topic == MathTopic.UNKNOWN
        assert p.needs_clarification is False
        assert p.variables == []

    def test_solution_model(self):
        from backend.models import Solution, SolutionStep
        step = SolutionStep(step_number=1, description="Compute", result="42")
        sol  = Solution(final_answer="42", steps=[step], confidence=0.95)
        assert sol.confidence == 0.95
        assert len(sol.steps) == 1

    def test_memory_record_auto_id(self):
        from backend.models import MemoryRecord
        r = MemoryRecord(session_id="s1", original_input="test")
        assert len(r.record_id) > 0


# ── Text Processor ────────────────────────────────────────────────────────────

class TestTextProcessor:
    def test_clean_whitespace(self):
        from backend.multimodal.text_processor import clean_text
        cleaned = clean_text("  hello   world  ")
        assert cleaned == "hello world"

    def test_empty_string(self):
        from backend.multimodal.text_processor import clean_text
        assert clean_text("") == ""

    def test_nonprintable_removed(self):
        from backend.multimodal.text_processor import clean_text
        result = clean_text("hello\x00world")
        assert "\x00" not in result


# ── Memory Store ──────────────────────────────────────────────────────────────

class TestMemoryStore:
    def setup_method(self):
        """Use an in-memory SQLite database for tests."""
        import tempfile, os
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.tmp.close()

    def teardown_method(self):
        import os
        try:
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_save_and_retrieve(self):
        from backend.memory.memory_store import MemoryStore
        from backend.models import MemoryRecord
        store = MemoryStore(db_path=self.tmp.name)
        rec = MemoryRecord(session_id="test-sess", original_input="quadratic eq")
        store.save(rec)

        retrieved = store.get_by_session("test-sess")
        assert retrieved is not None
        assert retrieved.original_input == "quadratic eq"

    def test_get_similar(self):
        from backend.memory.memory_store import MemoryStore
        from backend.models import MemoryRecord
        store = MemoryStore(db_path=self.tmp.name)

        for i in range(3):
            rec = MemoryRecord(
                session_id=f"sess-{i}",
                original_input="solve quadratic equation using formula",
            )
            store.save(rec)

        similar = store.get_similar("quadratic roots formula", limit=5)
        assert len(similar) >= 1

    def test_update_feedback(self):
        from backend.memory.memory_store import MemoryStore
        from backend.models import MemoryRecord, UserFeedback, FeedbackType
        store = MemoryStore(db_path=self.tmp.name)
        rec = MemoryRecord(session_id="fb-sess", original_input="test")
        store.save(rec)

        feedback = UserFeedback(
            session_id="fb-sess",
            feedback_type=FeedbackType.CORRECT,
        )
        store.update_feedback(feedback)
        updated = store.get_by_session("fb-sess")
        assert updated.user_feedback == "correct"


# ── Embeddings ────────────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_embed_returns_array(self):
        from backend.rag.embeddings import embed_texts, embed_query
        import numpy as np
        texts = ["algebra quadratic formula", "integration by parts"]
        vecs  = embed_texts(texts)
        assert vecs.shape[0] == 2
        assert vecs.shape[1] > 0

    def test_embed_query_shape(self):
        from backend.rag.embeddings import embed_query
        vec = embed_query("find roots of polynomial")
        assert len(vec.shape) == 1
        assert vec.shape[0] > 0


# ── FAISS Vector Store ────────────────────────────────────────────────────────

class TestVectorStore:
    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def test_add_and_search(self):
        from backend.rag.vector_store import FAISSVectorStore
        store = FAISSVectorStore(index_path=self.tmpdir)
        texts  = ["quadratic formula x = (-b ± sqrt(b²-4ac)) / 2a"]
        store.add_documents(texts, sources=["algebra.txt"], chunk_ids=["c001"])

        results = store.search("quadratic formula", top_k=1)
        assert len(results) == 1
        assert "quadratic" in results[0].content.lower()
        assert results[0].score > 0.0

    def test_empty_store_returns_empty(self):
        from backend.rag.vector_store import FAISSVectorStore
        import tempfile
        fresh_dir = tempfile.mkdtemp()
        store = FAISSVectorStore(index_path=fresh_dir)
        results = store.search("anything", top_k=5)
        assert results == []


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
