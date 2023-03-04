from flask import Flask, redirect, url_for, render_template, request, session, jsonify
import random
import string
import pandas as pd
import time
import nltk
from nltk import tree, ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import numpy as np
# from __future__ import print_function
# import time
# import sib_api_v3_sdk
# from sib_api_v3_sdk.rest import ApiException
# from pprint import pprint

# configuration = sib_api_v3_sdk.Configuration()
# configuration.api_key['api-key'] = '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'



app = Flask(__name__)
app.secret_key = "abc"

GREETING_INPUTS = ["Hi", "Hey", "Helloo", "Hellooo", "Greetings", "Greeting", "What's up", "What is up"]
GREETING_REPLYS = ["are you fine", "how you feel", "how is today!", "how is  your work", "are you fine"]
OUTOFCONTEXT = ["weather", "soccer", "universe", "animal", "dance", "game", "politics", "fire", "water"]
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
# subject = "My Subject"
# html_content = "<html><body><h1>This is my first transactional email </h1></body></html>"
# sender = {"name":"Team Unwind","email":"shwara99@gmail.com"}
# to = [{"email":"iphoneshwara@gmail.com","name":"Priya"}]
# reply_to = {"email":"replyto@domain.com","name":"John Doe"}
# headers = {"Some-Custom-Name":"unique-id-1234"}
# params = {"parameter":"My param value","subject":"New Subject"}
# send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(to=to, reply_to=reply_to, headers=headers, html_content=html_content, sender=sender, subject=subject)



#load csv file and get question list and define variables
df = pd.read_csv('question-edited.csv')
questions = df["Question"].to_list()
remaining = df["Question"].to_list()
stress ,notstress, n_asked_q = 0 , 0 , 0
previous_question = None
max_len =16

#lematization
def LemTokens(tokens):
    wordNetLem = WordNetLemmatizer()
    return [wordNetLem.lemmatize(token) for token in tokens]

def LemNormalize(text):
    punct = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(word_tokenize(text.lower().translate(punct)))

# vectorizer instance
vect = TfidfVectorizer(tokenizer = LemNormalize, stop_words = None)

# DistilBert tockenizer instance
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# load trained distilbert model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
model.load_weights('distilbert_model5.h5')


# predict probabilities for each classes 
def predict_proba(text):
    encodings = tokenizer([text], max_length=max_len, truncation=True, padding=True)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings))) 
    preds = model.predict(dataset.batch(1)).logits
    res = tf.nn.softmax(preds, axis=1).numpy()
    return res

# predict class -  stress and notstress
def classify(text):
    res = predict_proba(text)
    stress_category = tf.math.argmax(res[0], axis=-1).numpy()
    if stress_category==0:
        return "notstress"
    else: return "stress"


# select next best question to ask for maintatining the flow of conversation
def check_pairwise_similarity():
    remaining.append(previous_question)
    tfidf = vect.fit_transform(remaining)
    pairwise_similarity = tfidf * tfidf.T 
    arr = pairwise_similarity.toarray() 
    np.fill_diagonal(arr, np.nan) 
    input_idx = remaining.index(previous_question)
    result_idx = np.nanargmax(arr[input_idx]) 
    next_best_question =remaining[result_idx]  
    remaining.remove(previous_question)
    return next_best_question

# calculate stress percentage
def calculate_stress_percentage():
    global stress, notstress
    if stress > 0 and notstress>0:
        prc = stress * 100 / (stress+notstress)
        return prc
    else: return 0

# def send_alert():
    # if prc > 80%
        # try:
        #     api_response = api_instance.send_transac_email(send_smtp_email)
        #     pprint(api_response)
        # except ApiException as e:
        #     print("Exception when calling SMTPApi->send_transac_email: %s\n" % e)

#reset parameters after a counselling session
def reset_parameters():
    global stress, notstress, n_asked_q, previous_question
    remaining = df["Question"].to_list()
    previous_question = ""
    stress ,notstress, n_asked_q = 0 , 0 , 0
    # report = pd.DataFrame(columns=['Question', 'Answer', 'Stress_level'])

# update stress level count
def update_stress_count(stress_category):
    global stress, notstress
    if stress_category == "stress":stress += 1
    elif stress_category == "notstress":notstress += 1

@app.route("/chat/stress_prc")
def stress_percentage():
    prc = calculate_stress_percentage()
    # todo - return prc to your stress button action
    return jsonify({"stress" :prc})



@app.route("/chat", methods=["POST", "GET"])
def generate_reply():
    global previous_question,stress, notstress
    if request.method == "POST":
        answer = request.json["answer"]

        if answer.lower() == 'stop' or len(remaining)==0:
            prc = calculate_stress_percentage()
            question = "Let's finish the session.\nYour stress percentage is "+ str(prc)+ "%" +"\nHave a nice day!" 
            reset_parameters()
            return jsonify({"question":question})

        if answer.lower() in OUTOFCONTEXT:
            question = random.choice(questions) 
            question = "Sorry, you are talking out of context" +"\n" + question
            return jsonify({"question":question})

        #select next question
        if not previous_question:
            question = random.choice(questions) 
            previous_question = question
        else:
            question = check_pairwise_similarity()
            previous_question = question

        # remove question from remaining question list
        if question in remaining: remaining.remove(question)

        # classify answer
        text = question + answer
        stress_category = classify(text)

        # update stress level count
        update_stress_count(stress_category)

        return jsonify({"question":question})
 
    else:
        question = random.choice(questions) 
        previous_question = question

        # remove question from remaining question list
        if question in remaining: remaining.remove(question)

        # question = random.choice(GREETING_INPUTS) +",  \nI am a counselling bot.Let's start the session. " + "\n" + question
        question = "I am a counselling bot.Let's start the session. " + "\n" + question
        return jsonify({"question":question})


if __name__ == "__main__":
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=True)