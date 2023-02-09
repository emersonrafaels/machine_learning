from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

"""

bert-base-NER

Model description=
bert-base-NER is a fine-tuned BERT model that is ready to use 
for Named Entity Recognition and achieves state-of-the-art 
performance for the NER task. It has been trained to recognize 
four types of entities: 
location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC).

Specifically, this model is a bert-base-cased model 
that was fine-tuned on the English version of 
the standard CoNLL-2003 Named Entity Recognition dataset.

If you'd like to use a larger BERT-large model 
fine-tuned on the same dataset, 
a bert-large-NER version is also available.

Model: https://huggingface.co/dslim/bert-base-NER

"""

def init_model():

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    return nlp

def get_predict_string(model, text):

    model_results = model(text)

    return model_results

def get_predict_word(model, text):

    model_results = [model(value) for value in text.split(" ")]

    return model_results


def orchestra_predict(text_to_predict):

    model_nlp = init_model()
    
    print("RESULT ANALYSED STRING\n")
    ner_results_strings = get_predict_string(model=model_nlp, text=text_to_predict)

    for result in ner_results_strings:
        print(result)

    print("RESULT ANALYSED WORD\n")
    ner_results_word = get_predict_word(model=model_nlp, text=text_to_predict)

    for result in ner_results_word:
        print(result)

    return ner_results_strings, ner_results_word

# TEXT TO USE
text = "Emerson Vinicius is affiliated Mercia Ranniere DADDA Rafael and Edvaldo Carlos 12/02/2012 Validity RG Nacional"

# CALL MODEL
result_model_string, result_model_word = orchestra_predict(text_to_predict=text)