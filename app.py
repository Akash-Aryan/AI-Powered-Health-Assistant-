import streamlit as st
from transformers import pipeline
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import torch




# Initialize NLP components (Load once to improve performance)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined healthcare FAQs and responses
FAQS = {
    "What causes sneezing?": "Sneezing is often caused by irritants like dust, pollen, or allergens. It could also indicate a cold or flu.",
    "What are common flu symptoms?": "Common flu symptoms include fever, cough, body aches, and fatigue.",
    "How can I schedule an appointment?": "You can schedule an appointment by contacting your nearest clinic or using an online booking system.",
    "What should I do if I have a fever?": "Drink plenty of fluids, rest, and monitor your temperature. If it persists, consult a healthcare provider.",
    "How can I prevent the flu?": "Getting a flu shot, maintaining good hygiene, and avoiding close contact with sick individuals can help prevent the flu.",
    "What are the symptoms of COVID-19?": "Common symptoms include fever, cough, loss of taste or smell, difficulty breathing, and fatigue.",
    "How do I know if I have allergies or a cold?": "Allergies usually cause sneezing, itchy eyes, and nasal congestion, while a cold includes fever and body aches.",
    "What are the benefits of drinking water?": "Drinking water helps with digestion, improves skin health, regulates body temperature, and prevents dehydration.",
    "When should I see a doctor for a headache?": "If your headache is severe, persistent, or accompanied by vision problems, dizziness, or nausea, consult a doctor.",
    "How can I manage stress effectively?": "Exercise, meditation, deep breathing, and maintaining a healthy lifestyle can help reduce stress.",
    "What foods boost the immune system?": "Fruits, vegetables, nuts, and foods rich in vitamin C and zinc help strengthen the immune system.",
    "What are the signs of dehydration?": "Signs include dry mouth, dark urine, dizziness, fatigue, and extreme thirst.",
    "How much sleep do adults need?": "Adults should get 7-9 hours of quality sleep per night for optimal health.",
    "What are the common symptoms of diabetes?": "Symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing.",
    "What is high blood pressure and how can I manage it?": "High blood pressure, or hypertension, is when the blood exerts too much force against the artery walls. Managing it includes eating healthy, exercising, and reducing stress.",
    "How can I maintain good heart health?": "Regular exercise, a balanced diet, avoiding smoking, and managing stress contribute to heart health.",
    "What are the best ways to treat a minor burn?": "Run cool water over the burn, apply aloe vera or a clean bandage, and avoid popping blisters.",
    "What should I do if I get food poisoning?": "Stay hydrated, rest, and eat bland foods. If symptoms are severe, seek medical attention.",
    "What are the benefits of regular exercise?": "Exercise improves cardiovascular health, boosts mood, strengthens muscles, and helps with weight management.",
    "How often should I get a medical checkup?": "Annual health checkups are recommended to monitor overall health and catch potential issues early."
}

FAQS.update({
    "What is the normal body temperature?": "The normal body temperature for adults is around 98.6°F (37°C), but it can vary slightly.",
    "How can I lower my cholesterol levels?": "You can lower cholesterol by eating a healthy diet, exercising, and taking prescribed medications if needed.",
    "What are the symptoms of anemia?": "Anemia symptoms include fatigue, dizziness, pale skin, and shortness of breath.",
    "How can I prevent osteoporosis?": "Eating calcium-rich foods, exercising, and getting enough vitamin D can help prevent osteoporosis.",
    "What should I do if I get a deep cut?": "Apply pressure to stop the bleeding, clean the wound, and seek medical help if necessary.",
    "What are the warning signs of a stroke?": "Signs include sudden numbness, confusion, trouble speaking, vision problems, and severe headache.",
    "How can I improve my digestion?": "Eating fiber-rich foods, staying hydrated, and exercising can improve digestion.",
    "What are common causes of back pain?": "Poor posture, muscle strain, and underlying conditions like arthritis can cause back pain.",
    "How can I reduce my risk of heart disease?": "Maintaining a healthy diet, exercising regularly, and avoiding smoking can reduce heart disease risk.",
    "What are the benefits of yoga?": "Yoga helps improve flexibility, reduce stress, and enhance overall mental and physical well-being.",
    "How can I treat seasonal allergies?": "Avoiding allergens, taking antihistamines, and using nasal sprays can help manage allergies.",
    "What should I do if I feel dizzy often?": "Staying hydrated, avoiding sudden movements, and consulting a doctor for underlying causes are recommended.",
    "How can I relieve a sore throat?": "Drinking warm fluids, using lozenges, and gargling salt water can help soothe a sore throat.",
    "What are the early signs of cancer?": "Unexplained weight loss, lumps, persistent cough, and changes in moles can be early signs of cancer.",
    "What is the best way to treat acne?": "Keeping your skin clean, using non-comedogenic products, and consulting a dermatologist can help treat acne.",
    "What are the side effects of antibiotics?": "Common side effects include nausea, diarrhea, and possible allergic reactions.",
    "How do I know if I have food allergies?": "Food allergies often cause swelling, itching, digestive issues, and in severe cases, anaphylaxis.",
    "What is the best way to quit smoking?": "Nicotine replacement therapy, counseling, and behavioral changes can help quit smoking.",
    "What are common sleep disorders?": "Insomnia, sleep apnea, and restless legs syndrome are common sleep disorders.",
    "How can I boost my metabolism?": "Eating protein-rich foods, exercising, and staying hydrated can help boost metabolism."
})


# Precompute FAQ embeddings for better performance
faq_embeddings = {faq: semantic_model.encode(faq, convert_to_tensor=True) for faq in FAQS.keys()}

# Function to preprocess user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Function for semantic matching
def find_best_match(user_input):
    user_embedding = semantic_model.encode(user_input, convert_to_tensor=True)
    scores = {faq: util.pytorch_cos_sim(user_embedding, faq_emb).item() for faq, faq_emb in faq_embeddings.items()}
    best_match = max(scores, key=scores.get)
    return best_match, scores[best_match]

# Healthcare chatbot logic
def healthcare_chatbot(user_input):
    user_input = preprocess_input(user_input).lower()
    if not user_input:  # Handling empty input case
        return "Please enter a valid question."

    # Check for semantic match with FAQs
    best_match, score = find_best_match(user_input)
    if score > 0.7:  # Threshold for semantic similarity
        return FAQS[best_match]

    # Use QA model for general questions
    try:
        response = qa_pipeline(question=user_input, context=" ".join(FAQS.values()))
        return response['answer']
    except Exception as e:
        return f"Error processing your request: {e}"

# Main function for Streamlit app
def main():
    st.title("Akash Healthcare Chatbot")
    st.subheader("Your virtual healthcare assistant")

    user_input = st.text_input("How can I assist you today?", "")
    
    if st.button("Submit"):
        if user_input.strip():
            st.write("**User:** ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("**Healthcare Assistant:** ", response)
        else:
            st.warning("Please enter a query before submitting!")

    # Add FAQ section
    st.write("### Frequently Asked Questions:")
    for question in FAQS.keys():
        if st.button(question):
            st.write("**Answer:** ", FAQS[question])

if __name__ == "__main__":
    main()
