from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
# from main import generate_embedding
from embedding_utils import generate_embedding
import json
import uuid

# PENDING_DIR = Path(r"C:\Idea Projects\AI_Test_Assist\insights\pending")

# def save_pending_insight(
#     ticket_id: str,
#     step_number: int,
#     step_text: str,
#     selector: str,
#     module: str = "Unknown"
# ):
#     PENDING_DIR.mkdir(parents=True, exist_ok=True)

#     insight = {
#         "step": step_text,
#         "selector": selector,
#         "confidence": 1.0,                 # 🔒 HUMAN FEEDBACK = 1.0
#         "category": "failed",
#         "module": module,
#         "context": {
#             "module": module,
#             "action_type": "click",
#             "sequential_context": {}
#         },
#         "metadata": {
#             "ticket_id": ticket_id,
#             "step_number": step_number,
#             "timestamp": datetime.utcnow().isoformat(),
#             "issue_description": "selector corrected by user",
#             "tester_id": "ui_feedback",
#             "source": "tester_feedback",
#             "feedback_type": "failed"
#         }
#     }

#     filename = (
#         f"{ticket_id}_step{step_number}_"
#         f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
#         f"{uuid.uuid4().hex[:6]}.json"
#     )

#     file_path = PENDING_DIR / filename

#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(insight, f, indent=2)

#     return str(file_path)

def normalize_selector(selector: str) -> str:
    if not selector:
        return selector

    selector = selector.strip()

    # 🔥 FORCE valid CSS selector
    if not selector.startswith("["):
        selector = "[" + selector
    if not selector.endswith("]"):
        selector = selector + "]"

    return selector


router = APIRouter()

class SelectorFeedback(BaseModel):
    ticket_id: str
    step_number: int
    step_text: str
    corrected_selector: str
    module: str
    action_type: str
    status: str

# def generate_embedding(selector: str):
#     # Replace with your actual embedding logic
#     # return [0.1, 0.2, 0.3]
#     insight['embedding'] = generate_embedding(f"{step_text} {module}".strip())
    

@router.post("/api/selector-feedback")
def submit_selector_feedback(feedback: SelectorFeedback):
    normalized_selector = normalize_selector(feedback.corrected_selector)
    print(f"🔧 Normalized selector saved: {normalized_selector}")

    # embedding = generate_embedding(feedback.corrected_selector)
    embedding_text = f"{feedback.step_text} {feedback.module}".strip()
    embedding = generate_embedding(embedding_text)
    feedback_dir = Path("C:/AssureX/backend/insights/pending")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{feedback.ticket_id}_step{feedback.step_number}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    file_path = feedback_dir / file_name
    # with open(file_path, "w", encoding="utf-8") as f:
    #     json.dump({
    #         "step": feedback.step_text,
    #         "selector": feedback.corrected_selector,
    #         "confidence": 0.95,
    #         "category": "failed" if feedback.status == "FAILED" else "suspicious",
    #         "module": feedback.module,
    #         "context": {
    #             "action_type": feedback.action_type,
    #             "current_module": feedback.module
    #         },
    #         "metadata": {
    #             "ticket_id": feedback.ticket_id,
    #             "step_number": feedback.step_number,
    #             "timestamp": datetime.now().isoformat(),
    #             "issue_description": "selector not found" if feedback.status == "FAILED" else "manual correction",
    #             "source": "tester_feedback"
    #         },
    #         "embedding": embedding
    #     }, f, indent=2)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({
            "step": feedback.step_text,
            # "selector": feedback.corrected_selector,
            "selector": normalized_selector,
            "confidence": 0.95,
            "category": "failed",
            "module": feedback.module,
            "context": {
            "module": feedback.module,
            "action_type": "click",  # or use feedback.action_type if appropriate
            "sequential_context": {
                "previous_steps": [],
                "last_successful_action": None,
                "last_selector_used": None,
                "page_state_before_failure": "After step 1",
                "current_module": feedback.module
            }
        },
        "metadata": {
            "ticket_id": feedback.ticket_id,
            "step_number": feedback.step_number,
            "timestamp": datetime.now().isoformat(),
            "issue_description": "selector not found",
            "reasoning": "",
            "browser": "edge",
            "tester_id": "manual_feedback",
            "source": "tester_feedback",
            "feedback_type": "failed"
        },
        "embedding": embedding
    }, f, indent=2)
    return {"message": "Feedback saved"}