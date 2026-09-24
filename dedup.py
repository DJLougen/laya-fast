"""Request-level question dedup for LayaMLX.system_one.

Jev-style requests often repeat the same question under different ids (e.g. a
harness asking the same gate for several candidate actions, or retries merged
into one batch). The model forward is the dominant cost and is deterministic,
so identical questions only need one row in the batch.

dedup_system_one(agent, state, questions) groups question ids by their exact
(type, instructions, criteria) content, runs ONE agent.system_one call over the
unique questions, and fans the answers back out to every original id.

Contract:
  - answers[qid] is identical to what agent.system_one(state, questions) would
    return for that qid (same dict object content; duplicates share the answer).
  - usage.input_tokens reflects the deduplicated forward actually run
    (lower than a full call when duplicates exist). usage["dedup"] reports
    {"questions": N, "unique": M} so callers can see the saving.

Dedup key: (type, instructions, criteria) serialized with dict order preserved.
Two criteria dicts with the same pairs in different order are NOT merged —
option order changes the rendered sequence and can change the answer.
"""
import json
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import laya_api


class _SystemOneAgent(Protocol):
    """Anything exposing the Jev-shaped ``system_one`` request API."""

    def system_one(self, state: Any, questions: "laya_api.Questions") -> "laya_api.SystemOneResult": ...


def _qkey(q: "laya_api.QuestionDef") -> tuple[Any, Any, str]:
    """Canonical identity of a question's model-visible content."""
    crit = q.get("criteria")
    # Preserve mapping order: option order is part of the model input.
    if isinstance(crit, dict):
        crit_repr = json.dumps(list(crit.items()), ensure_ascii=False)
    else:
        crit_repr = json.dumps(crit, ensure_ascii=False)
    return (q.get("type"), q.get("instructions"), crit_repr)


def dedup_system_one(agent: _SystemOneAgent, state: Any, questions: "laya_api.Questions") -> "laya_api.SystemOneResult":
    """Like agent.system_one(state, questions) but runs one forward per unique
    (type, instructions, criteria) question and fans answers back to all ids."""
    if not questions:
        return {"model": "rl-agent", "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "dedup": {"questions": 0, "unique": 0}}}

    # First occurrence wins; rep qid is the id of the first question with
    # this exact (type, instructions, criteria) content.
    reps: dict[tuple[Any, Any, str], str] = {}   # key -> rep qid
    rep_of: dict[str, str] = {}                 # qid -> rep qid
    unique_questions: "laya_api.Questions" = {}
    for qid, q in questions.items():
        k = _qkey(q)
        rep = reps.get(k)
        if rep is None:
            rep = qid
            reps[k] = rep
            unique_questions[rep] = q
        rep_of[qid] = rep

    if len(unique_questions) == len(questions):
        result = agent.system_one(state, questions)
        result["usage"]["dedup"] = {"questions": len(questions),
                                    "unique": len(questions)}
        return result

    result = agent.system_one(state, unique_questions)
    # Fan out in the original question order.
    result["answers"] = {qid: result["answers"][rep_of[qid]] for qid in questions}
    result["usage"]["dedup"] = {"questions": len(questions),
                                "unique": len(unique_questions)}
    return result
