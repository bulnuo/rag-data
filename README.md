PACKAGES

pip install qdrant-client pytest numpy
pip install transformers
pip install sentence-transformers

TESTS

- test_vdb_insert.py - generate random vectors and store in the database. Tests 2 databases - Qdrant and Milvus

- test_embedder.py - vectorize random text sentences

- test_search.py - vectorize sentences, store in the database and search for matches


QUADRANT

docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant &
curl http://localhost:6333

  
MILVUS
pip install pymilvus
./standalone_embed.sh start

curl http://localhost:2379/health
docker logs milvus-standalone
./standalone_embed.sh stop
./standalone_embed.sh delete

