from collections import namedtuple
import math
import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from data_collator import DataCollatorForMultipleChoice
import time

from datasets import Dataset
from transformers import AutoTokenizer

def preprocess_function(examples, ending_names=["OptionA", "OptionB", "OptionC"]):
    # Repeat each first sentence three times to go with the three possibilities of second sentences.
    first_sentences = [[context] * 3 for context in examples["FalseSent"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["sent2"]
    second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]
    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])
    sentences = [first + " " + second for first, second in zip(first_sentences, second_sentences)]

    
    # Tokenize
    tokenized_examples = tokenizer(sentences, truncation=True)
    # Un-flatten
    return {k: [v[i:i+3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}

desc = "Just type in a sentence that you believe is wrong as well as three options \
for why it is wrong and we'll tell you which option actually makes sense."

st.title('Automated Common Sense Reasoning')
st.write(desc)

fs = st.text_input("False Sentence", "The cookie eats me.", key = "fs")

oa = st.text_input("Option A",  "Cookies are usually brown and I am tall.", key = "oa")
ob = st.text_input("Option B", "I like cookies.", key = "ob")
oc = st.text_input("Option C", "Cookies cannot eat humans.", key = "oc")

@st.cache(allow_output_mutation=True)
def get_data():
    return []


if st.button('Get Answer'):

    get_data().append({"FalseSent": fs,
        "OptionA": oa,
        "OptionB": ob,
        "OptionC": oc})
    
    choice_df = pd.DataFrame(get_data())
    choice_df['label'] = 0
    choice_df['sent2'] = ''

    # st.write(choice_df)

    dataset_test = Dataset.from_pandas(choice_df)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    encoded_dataset_sample = dataset_test.map(preprocess_function, batched=True)
    model = AutoModelForMultipleChoice.from_pretrained("artifacts")
    
    # try:
    #     model = AutoModelForMultipleChoice.from_pretrained("artifacts")
    # except:
    #     st.write("Something went wrong...")
    
    trainer = Trainer(
    model = model,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),)

    predictions = trainer.predict(encoded_dataset_sample).predictions
    pred_label = predictions.argmax(-1).astype(str)

    bar = st.progress(0)

    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.03)

    time.sleep(0.5)
    if pred_label.item(0) == '0':
        st.write('The correct answer is A.')
        # st.balloons()
        # st.metric(label="Correct Answer", value="A")
    elif pred_label.item(0) == '1':
        st.write('The correct answer is B.')
        # st.balloons()
        # st.metric(label="Correct Answer", value="A")
    elif pred_label.item(0) == '2':
        st.write('The correct answer is C.')
        # st.balloons()
        # st.metric(label="Correct Answer", value="A")
    else:
        st.write('Oops, something went wrong!')

# if not st.button('Reset'):
#     st.stop()

# st.metric(label, value, delta=None, delta_color="normal")

