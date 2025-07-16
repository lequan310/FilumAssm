import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from embedding import create_embedding
from milvus_connector import MilvusConnector

load_dotenv()

milvus_connector = MilvusConnector(
    uri=f"http://{os.getenv('MILVUS_HOST')}:{os.getenv('MILVUS_PORT')}",
    token=os.getenv("MILVUS_TOKEN"),
)


async def upsert_to_milvus(folder_path: str, collection_name: str):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist.")

    await milvus_connector.create_collection(collection_name)

    uploaded_data = []
    count = 0

    # Process all subdirectories
    for category_folder in folder_path.iterdir():
        if not category_folder.is_dir():
            continue

        category_name = category_folder.name
        print(f"Processing category: {category_name}")

        # Process all .md files in the category folder
        md_files = list(category_folder.glob("*.md"))

        if not md_files:
            print(f"No .md files found in {category_folder}")
            continue

        for md_file in md_files:
            print(f"Upserting file: {md_file.name}")

            # Read the markdown file content
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            data = {
                "id": count,
                "text": content,
                "dense": await create_embedding(content),
                "category": category_name,
            }
            uploaded_data.append(data)
            count += 1

    # Upsert all collected data to Milvus
    if uploaded_data:
        await milvus_connector.upsert_data(
            data=uploaded_data, collection_name=collection_name
        )
        print(
            f"Upserted {len(uploaded_data)} records to collection '{collection_name}'"
        )


async def main():
    folder_path = "data"
    collection_name = "default_collection"

    await upsert_to_milvus(folder_path, collection_name)

    # results = await milvus_connector.hybrid_search(
    #     query="We're struggling to collect customer feedback consistently after a purchase.",
    #     collection_name=collection_name,
    #     limit=2,
    # )
    # print([results[i]["entity"]["text"] for i in range(len(results))])


if __name__ == "__main__":
    asyncio.run(main())
