from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def semantic_similarity(prediction, answer):

    pred_emb = model.encode([prediction])
    ans_emb = model.encode([answer])

    score = cosine_similarity(
        pred_emb,
        ans_emb
    )[0][0]

    return score