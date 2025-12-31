# test_embeddings.py
from sentence_transformers import SentenceTransformer, util
m = SentenceTransformer("all-MiniLM-L6-v2")
emb = m.encode(["this is a test", "this is another test"], convert_to_tensor=True)
print("Embedding shape:", emb.shape)
print("Cosine similarity:", util.cos_sim(emb[0], emb[1]).item())
