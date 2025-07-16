from typing import Optional

import nest_asyncio
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    DataType,
    Function,
    FunctionType,
    WeightedRanker,
)

from embedding import create_embedding

nest_asyncio.apply()


class MilvusConnector:
    def __init__(
        self, uri: str, token: Optional[str] = None, db_name: Optional[str] = "default"
    ):
        self.uri = uri
        self.token = token
        self.client = AsyncMilvusClient(uri=uri, token=token, db_name=db_name)

    async def create_collection(self, collection_name: str):
        try:
            await self.client.drop_collection(collection_name)
        except Exception:
            pass

        # Create schema for the collection
        schema = AsyncMilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # Add fields to the schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=3072)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=8192,
            enable_analyzer=True,
        )
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # BM25 function for full text search
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)

        # Prepare index params
        index_params = AsyncMilvusClient.prepare_index_params()

        # Add indexes
        index_params.add_index(field_name="id", index_type="AUTOINDEX")
        index_params.add_index(
            field_name="dense", index_type="AUTOINDEX", metric_type="IP"
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )

        # Create the collection with the schema and indexes
        await self.client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )

    async def upsert_data(self, data: list, collection_name: str):
        await self.client.upsert(collection_name=collection_name, data=data)

    async def hybrid_search(
        self,
        query: str,
        collection_name: str,
        limit: int = 10,
        output_fields: Optional[list[str]] = None,
    ):
        vector = await create_embedding(query)

        dense_search_request = AnnSearchRequest(
            anns_field="dense", data=[vector], limit=limit, param={"nprobe": 10}
        )
        sparse_search_request = AnnSearchRequest(
            anns_field="sparse",
            data=[query],
            limit=limit,
            param={"drop_ratio_search": 0.2},
        )
        reqs = [dense_search_request, sparse_search_request]
        ranker = WeightedRanker(0.8, 0.2)

        results = await self.client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields,
        )

        return results[0]
