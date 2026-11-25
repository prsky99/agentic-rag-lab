"""별칭 원자 교체 (3.3절).

읽기는 별칭을 봅니다. 재색인은 새 컬렉션에 하고, 끝나면 별칭만 옮깁니다.
옮기는 순간은 원자적이므로 읽기가 끊기지 않습니다.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models


def swap_alias(client: QdrantClient, alias: str, new_collection: str) -> None:
    """별칭을 새 컬렉션으로 원자적으로 옮깁니다. 읽기는 끊기지 않습니다."""
    client.update_collection_aliases(change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)),
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=new_collection,
                                            alias_name=alias)),
    ])


def create_alias(client: QdrantClient, alias: str, collection: str) -> None:
    client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)),
    ])
