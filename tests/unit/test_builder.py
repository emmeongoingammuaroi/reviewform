"""Unit tests for the graph builder and conditional edges."""

from langgraph.graph import END

from app.agent.builder import should_continue_after_human


class TestConditionalEdges:
    def test_approved_routes_to_log_eval(self):
        """Approved review should route to log_eval node."""
        state = {"human_approved": True}
        assert should_continue_after_human(state) == "log_eval"

    def test_rejected_routes_to_review_code(self):
        """Rejected review should loop back to review_code."""
        state = {"human_approved": False}
        assert should_continue_after_human(state) == "review_code"

    def test_no_input_routes_to_end(self):
        """No human input (paused) should route to END."""
        state = {"human_approved": None}
        assert should_continue_after_human(state) == END

    def test_missing_key_routes_to_end(self):
        """Missing human_approved key should route to END (graph is paused)."""
        state = {}
        assert should_continue_after_human(state) == END
