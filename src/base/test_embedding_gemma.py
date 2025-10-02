import torch 
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("google/embeddinggemma-300M")

print(model)
print("T")