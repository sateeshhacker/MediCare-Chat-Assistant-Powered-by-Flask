# MediCare-Chat-Assistant-Powered-by-Flask

## Introduction

This project implements a simple chatbot using Python and Flask. The chatbot is trained to respond to user queries based on predefined patterns using natural language processing (NLP) techniques.

## Requirements

- Python 3.x
- Flask
- NLTK
- Keras
- NumPy

  ## Files

- `app.py`: Contains the main Flask application code for running the chatbot server.
- `training.py`: Contains the code for training the chatbot model using a dataset of intents.
- `data.json`: JSON file containing the intents and patterns for training the chatbot.
- `model.h5`: Pre-trained model file saved after training the chatbot.
- `texts.pkl`: Pickle file containing preprocessed text data used in training.
- `labels.pkl`: Pickle file containing the labels/classes used in training.

## Setup Instructions

### 1. Clone the Repository

```
git clone <repository_url>
cd medicare-chat-assistant
```

### 2. Set Up Virtual Environment

```
# Create a virtual environment (optional but recommended)
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Required Modules

```
pip install -r requirements.txt
```

### 4. Train the Model

Before running the chatbot, you need to train the model using the provided training script.

```
python training.py
```

This script preprocesses the data, creates a neural network model, and saves the trained model to a file (`model.h5`).

### 5. Run the Chatbot

After training the model, you can start the Flask application by running `app.py`.

```
python app.py
```

This will start the chatbot server locally, and you can interact with it through a web interface.

### 6. Interact with the Chatbot

Access the chatbot interface by opening a web browser and navigating to `http://localhost:5000`. You can then type in your queries and receive responses from the chatbot.

## Notes

- The chatbot model is trained on a dataset of intents stored in `data.json`. You can modify this file to add new intents or customize the responses.
- The chatbot uses a bag-of-words approach for text classification and a simple feedforward neural network architecture for training the model.
- Make sure to install the required Python packages listed in Requirements, before running the application.

