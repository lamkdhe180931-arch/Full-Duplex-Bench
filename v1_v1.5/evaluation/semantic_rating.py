import json
import os
import re


def _extract_json_payload(text):
    text = text or ""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in semantic rating response.")
    return text[start : end + 1]


def _load_payload(text):
    return json.loads(_extract_json_payload(text))


def _score(value):
    if value is None:
        return None
    value = float(value)
    value = max(0.0, min(5.0, value))
    return int(value) if value.is_integer() else value


def parse_answer_relevance_response(text):
    payload = _load_payload(text)
    return {
        "answer_relevance_rating": _score(
            payload.get("answer_relevance_rating", payload.get("rating"))
        ),
        "answer_relevance_analysis": str(
            payload.get("answer_relevance_analysis", payload.get("analysis", ""))
        ).strip(),
    }


def parse_interruption_semantic_response(text):
    payload = _load_payload(text)
    return {
        "previous_answer_rating": _score(payload.get("previous_answer_rating")),
        "previous_answer_analysis": str(payload.get("previous_answer_analysis", "")).strip(),
        "new_intent_rating": _score(payload.get("new_intent_rating")),
        "new_intent_analysis": str(payload.get("new_intent_analysis", "")).strip(),
        "old_context_leakage_score": _score(payload.get("old_context_leakage_score")),
        "old_context_leakage_analysis": str(payload.get("old_context_leakage_analysis", "")).strip(),
        "overall_semantic_rating": _score(payload.get("overall_semantic_rating")),
    }


def _generate_content(client, prompt):
    model_name = os.getenv("GEMINI_RATING_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text or ""


def rate_answer_relevance(client, task_name, user_request, answer_text, extra_context=""):
    prompt = f"""
Bạn là evaluator cho benchmark hội thoại full-duplex bằng tiếng Việt.
Hãy chấm câu trả lời của AI có liên quan và đúng trọng tâm với yêu cầu user không.

Task: {task_name}
Yêu cầu/câu hỏi của user:
{user_request}

Ngữ cảnh bổ sung:
{extra_context}

Câu trả lời của AI cần chấm:
{answer_text}

Thang điểm answer_relevance_rating từ 0 đến 5:
- 0: hoàn toàn không liên quan hoặc không có câu trả lời.
- 1: gần như không liên quan.
- 2: hơi liên quan nhưng thiếu trọng tâm.
- 3: có liên quan nhưng còn thiếu/lan man.
- 4: đúng trọng tâm và hữu ích.
- 5: rất đúng trọng tâm, đầy đủ, tự nhiên.

Chỉ trả về JSON hợp lệ, không markdown:
{{
  "answer_relevance_rating": 0,
  "answer_relevance_analysis": "nhận xét ngắn bằng tiếng Việt"
}}
"""
    raw = _generate_content(client, prompt)
    parsed = parse_answer_relevance_response(raw)
    parsed["raw_semantic_response"] = raw
    return parsed


def rate_interruption_semantics(
    client,
    context_request,
    interrupt_request,
    pre_interrupt_text,
    post_interrupt_text,
    full_output_text,
):
    prompt = f"""
Bạn là evaluator cho benchmark User Interruption trong hội thoại full-duplex bằng tiếng Việt.
Cần chấm riêng 2 lượt trả lời của AI: trước khi bị ngắt lời và sau khi user ngắt lời xong.

Câu hỏi/context ban đầu của user:
{context_request}

Câu ngắt lời/yêu cầu mới của user:
{interrupt_request}

Đoạn AI nói TRƯỚC khi user bắt đầu ngắt lời:
{pre_interrupt_text}

Đoạn AI nói SAU khi user ngắt lời xong:
{post_interrupt_text}

Toàn bộ transcript AI để đối chiếu:
{full_output_text}

Hãy chấm:
1. previous_answer_rating: đoạn trước interrupt có đang trả lời đúng câu hỏi ban đầu không.
2. new_intent_rating: đoạn sau interrupt có bám đúng yêu cầu mới không.
3. old_context_leakage_score: sau interrupt còn bị dính/lặp/bám ý cũ bao nhiêu, 0 là không dính, 5 là dính rất nặng.
4. overall_semantic_rating: tổng chất lượng semantic sau khi xét cả 2 lượt.

Mỗi điểm từ 0 đến 5. Chỉ trả về JSON hợp lệ, không markdown:
{{
  "previous_answer_rating": 0,
  "previous_answer_analysis": "nhận xét ngắn bằng tiếng Việt",
  "new_intent_rating": 0,
  "new_intent_analysis": "nhận xét ngắn bằng tiếng Việt",
  "old_context_leakage_score": 0,
  "old_context_leakage_analysis": "nhận xét ngắn bằng tiếng Việt",
  "overall_semantic_rating": 0
}}
"""
    raw = _generate_content(client, prompt)
    parsed = parse_interruption_semantic_response(raw)
    parsed["raw_semantic_response"] = raw
    return parsed
