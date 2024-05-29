#import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus

from .qdrant_utils import collection_exists
from .vectordb_interface import VectorDBInterface, MatchPoint

class QdrantVectorDB(VectorDBInterface):

    def __init__(self, host='localhost', port=6333, collection="default_collection", vector_size = 8):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection
        self.vector_size = vector_size
        print(f"Connected to Qdrant on {host}:{port}.")

        if (collection_exists(self.client,self.collection_name) == False):
            print(f"Creating new collection: {self.collection_name}, vector size: {self.vector_size}")
            self.collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
        else:
            print(f"Found existing collection: {self.collection_name}")
            self.client.get_collection(collection_name=self.collection_name)

    def recreate_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
        )

    def store_vectors(self, ids, vectors, payloads):
        payloads_as_dict = [{"text_to_embed": string} for string in payloads]
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads_as_dict
            )
        )
        print(f"Stored {len(vectors)} vectors in the collection.")

    def search(self, query_vector, top_k=3):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        matches = []
        for point in results[:top_k]:
            point = MatchPoint(point.id, point.score, payload=point.payload)
            matches.append(point)

        return matches 

    def retrieve_payload(self, id):
        results =  self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[id],
                    with_vectors=False,
                    with_payload=True
                )
        matches = []
        for point in results:
            point = MatchPoint(point.id, payload=point.payload)
            matches.append(point)

        if (len(matches) > 1):
            print(f"{len(matches)} are found with id = {id}")
        elif (len(matches) == 1):
            return matches[0]

    def retrieve_payload_by_PK(self, pk):
        return self.retrieve_payload(pk)

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)

    def collection_count(self):
        return self.client.count(
                    collection_name=self.collection_name 
                )

    def collection_list(self):
        collections = self.client.get_collections()

        print("Collections in Qdrant:")
        for collection in collections.collections:
            print("\t"+collection.name)
