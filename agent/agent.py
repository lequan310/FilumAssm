from datetime import datetime
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agent.state import State
from agent.tools import search_products


class PainPointAgent:
    def __init__(
        self,
        model: str | BaseChatModel,
        name: str = "PainPointAgent",
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self.name = name
        self.agent: CompiledStateGraph = create_react_agent(
            model=model,
            name=name,
            prompt=self.prompt,
            pre_model_hook=self.pre_model_hook,
            state_schema=State,
            tools=[search_products],
            checkpointer=checkpointer,
        )

    def prompt(self, state: State, config: RunnableConfig) -> list[AnyMessage]:
        system_prompt = f"""\
# Role
You are Fil, a product recommendation agent from Filum.ai. Your primary role is to identify and extract the pain point from the user input if available,analyze the user’s pain points in the business and propose a suitable Filum.ai’s AI product.

# Guidelines
- You MUST respond in the same language as the user.
- When the user presents a pain point, you should first identify, extract, and analyze the pain point and its category. Then, use your provided tools to find proper solutions to their problem and present them.
- Use only relevant tool results to answer the user’s question. Ignore irrelevant tool results.
- When presenting a solution, you should describe the product briefly and explain how it can help solve the user’s problem.
- Your response MUST be CONCISE and to the point, focusing on the product and how the proposed solution can help that exact pain point. Do NOT provide unnecessary information or details such as user's pain point, category, tools used, or any technical details.
- If you are unsure about something or need more information from the user, you may ask them for additional information.
- Do NOT mention anything related to the tools you are using, such as tool names, parameters, or any technical details.

# Additional Context
- Today’s date is {datetime.now().strftime("%A, %d/%m/%Y")}.
- Filum.ai is a Customer Experience and Customer Service Platform powered by Generative AI.\
"""

        return [{"role": "system", "content": system_prompt}] + state.get(
            "messages", []
        )

    async def pre_model_hook(self, state: State):
        return {
            "pain_point": state.get("pain_point", None),
            "category": state.get("category", None),
        }

    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None):
        return await self.agent.ainvoke(input, config=config)

    async def astream(self, input: dict, config: Optional[RunnableConfig] = None):
        async for chunk, metadata in self.agent.astream(
            input,
            config=config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content
