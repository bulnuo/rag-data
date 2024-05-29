import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Adds the project root to sys.path
print("Added project root to sys.path")


from app.pipeline.embedders.sentence_handler import SentenceHandler

embeddings = []
sentences = [
    "The cat lazily stretched out on the windowsill, basking in the warm afternoon sun.",
    "A sudden gust of wind sent the leaves swirling through the air in a chaotic dance.",
    "She found an old, dusty book in the attic, filled with forgotten stories and memories.",
    "The aroma of freshly baked bread filled the kitchen, making everyone's mouths water.",
    "He couldn't believe his eyes when he saw the vibrant colors of the sunset over the ocean.",
    "The little girl giggled as she chased after the butterflies in the flower-filled meadow.",
    "With a determined look, the athlete crossed the finish line, breaking the previous record.",
    "The old man sat on the park bench, feeding the pigeons and reminiscing about his youth.",
    "The bustling market was alive with the sounds of haggling vendors and curious shoppers.",
    "As the rain poured down, the couple huddled under a single umbrella, laughing at the storm."
]

def test_embedder():
    
    embedder = SentenceHandler()

    for sentence in sentences:
        embedding = embedder.embed_texts(sentence)
        embeddings.append(embedding)
        print(f"Sentence: {sentence}")
        print(f"Embedding: {embedding[:5]} ...")  


if __name__ == "__main__":
    test_embedder()
   
