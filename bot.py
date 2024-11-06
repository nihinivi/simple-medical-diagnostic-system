import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

data = pd.read_csv('data.csv')[['Disease', 'Cleaned_Symptoms']]
model = SentenceTransformer('all-MiniLM-L6-v2')

data['Symptom_Embedding'] = data['Cleaned_Symptoms'].apply(lambda x: model.encode(x, convert_to_tensor=True))

def diagnose_symptoms(user_symptom):
    user_embedding = model.encode(user_symptom, convert_to_tensor=True)
    similarity_scores = data['Symptom_Embedding'].apply(lambda x: util.cos_sim(user_embedding, x).item())
    best_match_index = similarity_scores.idxmax()
    return data.loc[best_match_index, 'Disease']

user_input = "sharp abdominal pain and tenderness"
predicted_disease = diagnose_symptoms(user_input)
print(f"Predicted Disease: {predicted_disease}")
