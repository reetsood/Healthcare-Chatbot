import pickle
import re
import pandas as pd

# === Load ML model and vectorizer ===
with open('chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('symptom_map.pkl', 'rb') as f:
    symptom_map = pickle.load(f)

# Load CSVs for description, severity, and prescription
desc_df = pd.read_csv("C:/Users/sooda/Downloads/Healthcare_dataset/symptom_Description.csv")
pres_df = pd.read_csv("C:/Users/sooda/Downloads/Healthcare_dataset/symptom_precaution.csv")

# Normalize input text by replacing known phrases with underscored versions
def normalize_input(text, symptom_map):
    text = text.lower()
    for phrase, underscored in symptom_map.items():
        text = re.sub(rf'\b{re.escape(phrase)}\b', underscored, text)
    return text

print("🩺 Hi, I'm your health assistant. Tell me how you're feeling.")
collected_input = ""
while True:
    user_input = input("You: ")
    if user_input.lower().strip() in ['no', 'no more', 'that’s it', 'nothing else', "that's everything"]:
        break
    collected_input += " " + user_input
    print("Bot: Got it. Anything else you're feeling? Say 'no more' to finish.")

# --- Preprocess ---
cleaned_input = normalize_input(collected_input, symptom_map)

# --- Prediction ---
user_vector = vectorizer.transform([cleaned_input])
probs = model.predict_proba(user_vector)[0]
top_n = probs.argsort()[-3:][::-1]
predicted_diseases = [model.classes_[i] for i in top_n]

print("\n🤖 Based on what you've told me, here are the top 3 possible conditions:\n")
for i in top_n:
    print(f"🔹 {model.classes_[i]} ({probs[i]*100:.2f}%)")

# --- Descriptions of diseases ---
print("\n📋 Descriptions of these conditions:")
for disease in predicted_diseases:
    desc_row = desc_df[desc_df['Disease'].str.lower() == disease.lower()]
    if not desc_row.empty:
        print(f"\n🔸 {disease}: {desc_row.iloc[0]['Description']}")
    else:
        print(f"\n🔸 {disease}: No description available.")


# Show prescriptions/precautions based on predicted diseases
print("\n🛡️ Suggested Precautions for Predicted Diseases:")
for disease in predicted_diseases:
    presc_row = pres_df[pres_df['Disease'].str.lower() == disease.lower()]
    if not presc_row.empty:
        precautions = presc_row.iloc[0].drop('Disease').dropna().tolist()
        print(f"\n🔸 {disease}:")
        for p in precautions:
            print(f"   💊 {p}")
    else:
        print(f"\n🔸 {disease}: No precautions found.")
