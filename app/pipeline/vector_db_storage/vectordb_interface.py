
class MatchPoint:
   def __init__(self, id, score=0, payload=0, pk=0, vector=0):
      self.id = id
      self.score = score
      self.pk = pk
      self.payload = payload
      self.vector = vector

class VectorDBInterface:

  def __init__(self):
    self.dummy = 0

  def store_vectors(self, ids, vectors, payloads):
    raise NotImplementedError
  
  def search(self, query_vector, top_k):
    raise NotImplementedError

  def retrieve_payload(self, id):
    raise NotImplementedError

  def retrieve_payload_by_PK(self, pk):
    raise NotImplementedError

