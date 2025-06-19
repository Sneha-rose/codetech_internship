import json
import random
import string
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from transformers import pipeline

# Load environment variables
load_dotenv()

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLP components
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
try:
    with open('intents.json') as file:
        data = json.load(file)
    print("✅ Successfully loaded intents.json")

    # Validate JSON structure
    if 'intents' not in data or not isinstance(data['intents'], list):
        print("❌ Error: Invalid structure in intents.json (Expected 'intents' key with a list)")
        exit(1)

except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"❌ Error loading intents.json: {str(e)}")
    exit(1)

# Preprocess text function
def preprocess_text(text):
    """Tokenize, lemmatize, and clean text."""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Prepare data for classifiers
intent_patterns = []
tags = []
responses = {}

# Iterate through intents and check validity
for intent in data['intents']:
    if isinstance(intent, dict) and 'tag' in intent and 'patterns' in intent:
        tag = intent['tag']
        tags.append(tag)
        responses[tag] = intent.get('responses', ["I'm not sure how to respond to that."])

        if isinstance(intent['patterns'], list):
            for pattern in intent['patterns']:
                processed = preprocess_text(pattern)
                intent_patterns.append((processed, tag))
        else:
            print(f"⚠️ Warning: 'patterns' for intent '{tag}' is not a list, skipping.")
    else:
        print("⚠️ Warning: Skipping malformed intent entry.")

# Ensure at least one valid training pattern exists
if not intent_patterns:
    print("❌ Error: No valid training patterns found in intents.json. Check the file structure.")
    exit(1)

# Traditional TF-IDF setup
texts, labels = zip(*intent_patterns)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Transformer setup
transformer_model = "facebook/bart-large-mnli"
#classifier = pipeline(
 #  "zero-shot-classification",
  #  model="typeform/distilbert-base-uncased-mnli",
   #device=-1  # CPU
#)

def predict_intent(user_input):
    """Hybrid intent classification with fallback mechanism"""
    try:
        transformer_result = classifier(
            user_input, 
            candidate_labels=tags,
            multi_label=False
        )
        top_intent = transformer_result['labels'][0]
        confidence = transformer_result['scores'][0]
        
        if confidence > 0.7:
            return top_intent, confidence, "transformer"
    except Exception as e:
        #print(f"⚠️ Transformer error: {str(e)}")
        confidence = 0.0
    
    # Fallback to TF-IDF
    processed_input = preprocess_text(user_input)
    input_vec = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vec, X)
    
    if similarities.size > 0:
        max_index = np.argmax(similarities)
        tfidf_confidence = similarities[0, max_index]
        
        if tfidf_confidence > 0.3:
            return labels[max_index], tfidf_confidence, "tfidf"
    
    return None, max(confidence, 0.0), "fallback"

def get_response(intent_tag):
    """Get response for identified intent"""
    return random.choice(responses.get(intent_tag, ["I'm not sure how to respond to that."]))

def handle_unknown_query(user_input):
    """Handle unrecognized queries"""
    fallbacks = [
        "I'm still learning about that. Could you ask something else?",
        "That's an interesting question. Let me find more information.",
        "I'm not sure I understand. Could you rephrase your question?"
    ]
    return random.choice(fallbacks)

# Context tracking
conversation_history = []
CONTEXT_WINDOW = 3

def update_context(user_input, bot_response):
    """Maintain conversation context"""
    conversation_history.append((user_input, bot_response))
    if len(conversation_history) > CONTEXT_WINDOW:
        conversation_history.pop(0)

# Main chat loop
print("\n" + "="*50)
print("🤖 AI Business Assistant Initialized Successfully!")
print("="*50)
print("\nI can help with:\n- Order status\n- Product info\n- Business hours\n- Returns & policies")
print("- Account support\n- Technical issues\n- And more!")
print("\nType 'quit' to exit or 'history' to see context\n")

while True:
    try:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("\n🤖 Thank you for chatting! Have a great day!")
            break
            
        if user_input.lower() == 'history':
            print("\nConversation History:")
            for i, (user, bot) in enumerate(conversation_history[-CONTEXT_WINDOW:]):
                print(f"[{i+1}] You: {user}")
                print(f"    Bot: {bot}")
            print()
            continue
        
        # Handle contextual follow-ups
        if conversation_history:
            last_response = conversation_history[-1][1].lower()
            
            # Context: Hours -> Weekend hours
            if "hour" in last_response and any(word in user_input.lower() for word in ["weekend", "saturday", "sunday"]):
                response = "Weekend Hours: 10 AM to 3 PM"
                print(f"🤖 {response}")
                update_context(user_input, response)
                continue
                
            # Context: Returns -> Defective products
            if "return" in last_response and "defective" in user_input.lower():
                response = "For defective items: Contact support@company.com for expedited replacement"
                print(f"🤖 {response}")
                update_context(user_input, response)
                continue
        
        # Process input
        intent, confidence, method = predict_intent(user_input)
        
        response = get_response(intent) if intent else handle_unknown_query(user_input)
        print(f"🤖 {response}")
        update_context(user_input, response)
            
    except KeyboardInterrupt:
        print("\n\n🤖 Session ended by user. Goodbye!")
        break
    except Exception as e:
        print(f"🤖 ⚠️ Error processing request: {str(e)}")
        print("🤖 Please try again or rephrase your question")