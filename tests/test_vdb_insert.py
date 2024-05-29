import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Adds the project root to sys.path
print("Added project root to sys.path")


import pytest
import numpy as np
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams

from app.pipeline.vector_db_storage.vectordb_manager import VectorDBManager

NUMBER_OF_VECTORS=100
VECTOR_SIZE=10
BATCH_SIZE=20

vector_id = 0

def generate_ids(num):
    # Note, qdrant only allows int or uuids as vector ids

    # uncommnet for uuids
    # return [str(uuid.uuid4()) for _ in range(num)]

    # uncommnet for integers
    global vector_id
    return [(vector_id := vector_id + 1) for _ in range(num)]

def generate_random_vectors(num, size=10):
    return [np.random.rand(size).tolist() for _ in range(num)]

def test_qdrant():
    print("### Test Qdrant insert ###")

    # Generate random data
    list_of_id_lists = []
    list_of_value_lists = []

    for i in range(0, NUMBER_OF_VECTORS, BATCH_SIZE):
        list_of_id_lists.append(generate_ids(BATCH_SIZE))
        list_of_value_lists.append(generate_random_vectors(BATCH_SIZE, VECTOR_SIZE))
        print(f"Processed batch {i // BATCH_SIZE + 1}")

    #
    # Store vectors
    #

    database = VectorDBManager().get(database_type='qdrant',
                        collection_name='test',
                        vector_size=10)
    print(database)

    for idx, (id_list, vector_list) in enumerate(zip(list_of_id_lists, list_of_value_lists)):
        print(f"Storing batch {idx + 1}:")

        database.store_vectors(
            id_list,
            vector_list,
            payloads=vector_list             # for testing only
        )

    #
    # Baseline query
    #

    print("\n=== QUERY 1 ===")
    query1 = list_of_value_lists[1][1]

    print("Query 1 text:", query1)

    results = database.search(
        query_vector=query1,
        top_k=3
    )

    print("\nQUERY 1 RESULTS")
    print("\nTop Similar Vectors:\n")

    for point in results[:3]:
        print(f"ID: {point.id}, Score: {point.score} \nPAYLOAD: {point.payload} \n")

    match_point = results[0]
    assert 0.9 <= match_point.score <= 1.1, "Search did not match the query vector"

    print("\n=== Individual Lookups by ID ===\n")
    point = database.retrieve_payload(match_point.id)
    if point: print(f"Point ID: {point.id}, PAYLOAD: {point.payload}")
    assert point.payload['text_to_embed'] == match_point.payload['text_to_embed'], "Retrieved and searched vectors did not match"



def test_milvus():
    print("### Test Milvus insert ###")

    # Generate random data
    list_of_id_lists = []
    list_of_value_lists = []
    list_of_payload_lists = []

    for i in range(0, NUMBER_OF_VECTORS, BATCH_SIZE):
        list_of_id_lists.append(generate_ids(BATCH_SIZE))
        list_of_value_lists.append(generate_random_vectors(BATCH_SIZE, VECTOR_SIZE))
        list_of_payload_lists.append("dummy payload " + str(i))
        print(f"Processed batch {i // BATCH_SIZE + 1}")

    #
    # Store vectors
    #

    database = VectorDBManager().get(database_type='milvus',
                        collection_name='test',
                        vector_size=10)
    print(database)
    
    for idx, (id_list, vector_list, payload_list) in enumerate(zip(list_of_id_lists, list_of_value_lists, list_of_payload_lists)):
        print(f"Storing batch {idx + 1}:")

        database.store_vectors(
            id_list,
            vector_list,
            payloads=None
        )

    #
    # Baseline query
    #

    print("\n=== QUERY 1 ===")
    query1 = list_of_value_lists[1][1]

    print("Query 1 text:", query1)

    results = database.search(
        query_vector=query1,
        top_k=3
    )

    print("\nQUERY 1 RESULTS")
    print("\nTop Similar Vectors:\n")

    for point in results[:3]:
        print(f"ID: {point.id}, PK: {point.pk}, Score: {point.score} \nVector: {point.vector} \n")

    match_point = results[0]
    assert match_point.score == 0, "Search did not match the query vector"

    print("\n=== Individual Lookups by ID ===\n")
    results = database.retrieve_payload(match_point.id)
    for point in results:
        print(f"ID: {point.id}, PK: {point.pk} \nVector: {point.vector} \n")
    
    database.delete_collection()

if __name__ == "__main__":
    test_qdrant()
    test_milvus()

