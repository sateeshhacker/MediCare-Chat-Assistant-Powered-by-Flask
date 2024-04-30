import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

nltk.download('popular')
lemmatizer = WordNetLemmatizer()

# Load the trained model and other necessary data
model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

conversation_state = {}

def check_recognizable_words(message, words):
    # Tokenize the message
    message_words = nltk.word_tokenize(message)
    # Lemmatize and convert to lowercase
    message_words = [lemmatizer.lemmatize(word.lower()) for word in message_words]
    
    # Check if any word in the message is present in the model's vocabulary
    for word in message_words:
        if word in words:
            return True
    return False
 
    #-----------------------------HOSPITAL-------------------------------
def chatbot_response(message):
    global conversation_state 
    # Check if the message contains the word "hospital"
    if 'hospital' in message.lower():
        # Set conversation state to prompt for location
        conversation_state['prompt_for_location'] = True
        return "Sure, I can help you with hospitals. Please provide the location."
    
    elif 'prompt_for_location' in conversation_state and conversation_state['prompt_for_location']:
        # Set the location based on the user's message
        conversation_state['location'] = message.strip()
        del conversation_state['prompt_for_location']  # Remove the prompt flag
        
        # Check if the location is "punjab"
        if 'punjab' in conversation_state['location'].lower():
            hospitals = [
                "AJIT HOSPITAL - Green Avenue road, 309, circular road block market Ranjit Avenue",
                "AKASHDEEP HOSPITAL - Majitha Road, Opposite M.B Poly-technical College",
                "AMANDEEP HOSPITAL",
                "Amritsar Sewa Samiti Hospitals",
                "APEX HOSPITAL 1",
                "BAJWA HOSPITAL",
                "BIALA ORTHOPAEDICS AND MULTISPECIALITY HOSPITAL",
                "C.H.C Majitha"
            ]
            # Construct the response with hospital names colored blue
            response = "Sure, here are some hospitals in Punjab:<br>"
            for i, hospital in enumerate(hospitals):
                # Set the color of each hospital name to blue
                response += f'<span style="color: blue;">{i + 1}. {hospital}</span><br>'
            
            return response
        else:
            return "Sorry, I only have information for hospitals in Punjab."
    
    else:
        # Check if the message contains recognizable words
        if not check_recognizable_words(message, words):
            return "Sorry, I couldn't recognize your input. Please try again."
        
        # Use the existing chatbot response logic
        ints = predict_class(message, model)
        res = getResponse(ints, intents)
        return res

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()
