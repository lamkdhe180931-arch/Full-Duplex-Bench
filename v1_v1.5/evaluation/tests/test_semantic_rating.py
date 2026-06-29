import os
import sys
import unittest


EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from semantic_rating import (  # noqa: E402
    parse_answer_relevance_response,
    parse_interruption_semantic_response,
)


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


if __name__ == "__main__":
    unittest.main()
