from .qdrant import QdrantVectorDB
from .milvus import MilvusVectorDB

class VectorDBManager:
    def __init__(self):
        pass

    def get(self, database_type ='qdrant',collection_name='default_collection',vector_size=8):

        if database_type == 'qdrant':
            return QdrantVectorDB(collection=collection_name,vector_size=vector_size)
        elif database_type == 'milvus':
            return MilvusVectorDB(collection=collection_name,vector_size=vector_size)
        else:
            raise ValueError("Unsupported database type")


    '''
    def store_vectors(self, ids, vectors, payloads):    
        self.database.store_vectors(ids, vectors, payloads)

    def search(self, query_vector, top_k=3):
        return self.database.search(query_vector, top_k)

    def retrieve_payload(self, id):
        return self.database.retrieve_payload(id)

    def retrieve_payload_by_PK(self, pk):
        return self.database.retrieve_payload_by_PK(pk)
    '''