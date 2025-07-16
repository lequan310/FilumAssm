from typing import Literal, Optional

from langgraph.prebuilt.chat_agent_executor import AgentState

CATEGORY = Literal[
    "ai_automation", "customer_360", "customer_service", "insights", "voice_of_customer"
]


class State(AgentState):
    use_case: Optional[str]
    category: Optional[CATEGORY]
    relevant_documents: Optional[list[str]] = None
