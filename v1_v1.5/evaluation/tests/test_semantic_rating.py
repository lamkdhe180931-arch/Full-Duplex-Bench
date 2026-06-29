import os
import sys
import unittest


EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from semantic_rating import (  # noqa: E402
    parse_answer_relevance_response,
    parse_interruption_semantic_response,
    safe_rate_answer_relevance,
    safe_rate_interruption_semantics,
)


class BrokenModels:
    def generate_content(self, model, contents):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


class BrokenClient:
    models = BrokenModels()


class SemanticRatingTest(unittest.TestCase):
    def test_parse_answer_relevance_response_accepts_json_code_fence(self):
        parsed = parse_answer_relevance_response(
            """
            ```json
            {
              "answer_relevance_rating": 4,
              "answer_relevance_analysis": "Trả lời đúng trọng tâm."
            }
            ```
            """
        )

        self.assertEqual(parsed["answer_relevance_rating"], 4)
        self.assertEqual(parsed["answer_relevance_analysis"], "Trả lời đúng trọng tâm.")

    def test_parse_interruption_semantic_response_returns_three_scores(self):
        parsed = parse_interruption_semantic_response(
            """
            {
              "previous_answer_rating": 4,
              "previous_answer_analysis": "Đang trả lời đúng câu hỏi ban đầu.",
              "new_intent_rating": 5,
              "new_intent_analysis": "Đã chuyển đúng sang yêu cầu mới.",
              "old_context_leakage_score": 1,
              "old_context_leakage_analysis": "Chỉ còn nhắc nhẹ ngữ cảnh cũ.",
              "overall_semantic_rating": 4
            }
            """
        )

        self.assertEqual(parsed["previous_answer_rating"], 4)
        self.assertEqual(parsed["new_intent_rating"], 5)
        self.assertEqual(parsed["old_context_leakage_score"], 1)
        self.assertEqual(parsed["overall_semantic_rating"], 4)

    def test_safe_answer_relevance_returns_na_when_api_quota_fails(self):
        parsed = safe_rate_answer_relevance(
            BrokenClient(),
            task_name="Smooth Turn Taking",
            user_request="Câu hỏi user",
            answer_text="Câu trả lời AI",
        )

        self.assertIsNone(parsed["answer_relevance_rating"])
        self.assertIn("429 RESOURCE_EXHAUSTED", parsed["semantic_error"])

    def test_safe_interruption_semantics_returns_na_when_api_quota_fails(self):
        parsed = safe_rate_interruption_semantics(
            BrokenClient(),
            context_request="Câu hỏi cũ",
            interrupt_request="Câu hỏi mới",
            pre_interrupt_text="Trả lời cũ",
            post_interrupt_text="Trả lời mới",
            full_output_text="Trả lời cũ rồi mới",
        )

        self.assertIsNone(parsed["previous_answer_rating"])
        self.assertIsNone(parsed["new_intent_rating"])
        self.assertIsNone(parsed["old_context_leakage_score"])
        self.assertIn("429 RESOURCE_EXHAUSTED", parsed["semantic_error"])


if __name__ == "__main__":
    unittest.main()
