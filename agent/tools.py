import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from agent.state import CATEGORY
from milvus_connector import MilvusConnector

load_dotenv()


milvus_connector = MilvusConnector(
    uri=f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}",
    token=os.getenv("MILVUS_TOKEN"),
)


@tool
async def search_products(
    use_case: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    category: Optional[CATEGORY] = None,
):
    """Search for products based on the use case and category.

    Args:
        use_case (str): The use case for which to find products. Generate a use case based on the user's pain point.
        category (CATEGORY): The category of the product. Leave it empty if you want to search across all categories.
    """

    results = await milvus_connector.hybrid_search(
        query=use_case,
        collection_name="default_collection",
        output_fields=["text", "category"],
    )

    results = [result["entity"] for result in results]
    if category:
        # Filter results by category if provided
        results = [
            result["text"] for result in results if result["category"] == category
        ]
    else:
        results = [result["text"] for result in results]
    results = results[:2]

    if not results:
        message = "No products found for the given use case and category."
    else:
        message = (
            f"Here are some products that match your use case '{use_case}' in the category '{category}':\n\n"
            + "\n-------------------------------\n".join(results)
        )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=message,
                    tool_call_id=tool_call_id,
                )
            ],
            "pain_point": use_case,
            "category": category,
        }
    )
