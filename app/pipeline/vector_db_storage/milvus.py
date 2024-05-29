import numpy as np
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

from .vectordb_interface import VectorDBInterface, MatchPoint

class MilvusVectorDB(VectorDBInterface):

    def __init__(self, host='localhost', port=19530, collection="default_collection", vector_size = 8):

        self.client = connections.connect("default", host=host, port=port)
        self.collection_name = collection
        self.vector_size = vector_size
        print(f"Connected to Milvus on {host}:{port}.")

        fields = [
            FieldSchema(name="pks", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="ids", dtype=DataType.INT32),
            #FieldSchema(name="payloads", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=vector_size)
        ]
        schema = CollectionSchema(fields, "simple schema")

        if (utility.has_collection(self.collection_name) == False):
            self.collection = Collection(self.collection_name, schema, consistency_level="Strong")
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }

            self.collection.create_index("embeddings", index)
            self.collection.flush()
            print(f"Created collection {self.collection_name} and index")
        else:
            self.collection = Collection(self.collection_name, schema, consistency_level="Strong")
            print(f"Collection indexes {self.collection.indexes}")
        self.collection.flush()

    def store_vectors(self, ids, vectors, payloads):
        entities = [ids, vectors]
        length = len(ids)

        self.collection.load()
        insert_result = self.collection.insert(entities)

        print(f"Inserted {length} entities. Total number of entities is {self.collection.num_entities}")
        print(utility.index_building_progress(self.collection_name))
        self.collection.flush()
    
    def search(self, query_vector, top_k=3):
        self.collection.load()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result = self.collection.search([query_vector], "embeddings", search_params, limit=5, output_fields=["ids","pks","embeddings"])

        print("Top similar vectors:")
        matches = []

        for points in result:
           for point in points:
               id = point.entity.get('ids')
               pk = point.entity.get('pks')
               embeddings = point.entity.get('embeddings')
               point = MatchPoint(id, point.score, pk=pk, vector=embeddings)
               matches.append(point)

        self.collection.release()
        return matches

    def retrieve_payload(self, id):
        self.collection.load()
        results = self.collection.query(expr=f"ids == {id}", output_fields=["ids","pks","embeddings"])

        matches = []
        for point in results:
            id = point['ids']
            vector = point['embeddings']
            pk = point['pks']
            point = MatchPoint(id, pk=pk, vector=vector)
            matches.append(point)

        self.collection.release()

        return matches

    def retrieve_payload_by_PK(self, pk):
        self.collection.load() 
        results = self.collection.query(expr=f"pks == {pk}", output_fields=["ids","embeddings","pks"])

        matches = []
        for point in results:
            id = point['ids']
            vector = point['embeddingss']
            pk = point['pks']
            point = MatchPoint(id, vector=vector, pk=pk)
            matches.append(point)

        self.collection.release()

        if (len(matches) > 1):
            print(f"WARNING:Multiple matches - {len(matches)} - are found with PK = {pk}.")
        elif (len(matches) == 1):
            return matches[0]
        

    def delete_collection(self):
        utility.drop_collection(self.collection_name)
