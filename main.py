import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from agent.agent import PainPointAgent
from agent.llm import create_llm


async def main():
    llm = create_llm()
    memory_saver = InMemorySaver()
    agent = PainPointAgent(model=llm, checkpointer=memory_saver)
    thread_id = "default_thread"

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        print("Agent: ", end="", flush=True)
        async for chunk in agent.astream(
            input={"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": thread_id}},
        ):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
