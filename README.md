# Healthcare-Chatbot

This project is a machine learning-based chatbot that predicts diseases based on symptoms provided by the user. It is built using Python, scikit-learn, pandas, and natural language processing techniques. The chatbot can analyze multiple symptoms and return the most probable disease with high accuracy.

---

## 📌 Key Features

- Accepts symptoms via natural language input
- Predicts the most likely disease based on trained data
- Preprocesses symptom data with regex and normalization
- Uses TF-IDF vectorization for feature extraction
- Trained using Multinomial Naive Bayes with calibrated probabilities
- Extensible to include descriptions and prescriptions

---

## 🗃️ Dataset Used

The model is trained on a dataset (`dataset.csv`) containing:
- Multiple symptoms per patient (Symptom_1, Symptom_2, ..., Symptom_N)
- Corresponding disease for each symptom set

Additional CSV files (optional for extension):
- `symptom_description.csv` – Maps symptoms to descriptions
- `symptom_prescription.csv` – Maps symptoms to suggested prescriptions






## 🏗️ Project Structure

