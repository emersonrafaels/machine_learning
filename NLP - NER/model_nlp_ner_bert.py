from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

def init_model():

    # parameters
    model_name = "ner-bert-large-cased-pt"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def tokenizer_text(tokenizer, text):

    # tokenization
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
    tokens = inputs.tokens()

    return inputs, tokens

def get_predict(model, tokens, inputs):

    # get predictions
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)

    # print predictions
    for token, prediction in zip(tokens, predictions[0].numpy()):
        print((token, model.config.id2label[prediction]))

    return predictions

def orchestra_predict(text):

    # INIT MODEL
    model_nlp, tokenizer = init_model()

    # PREPROCESSING TEXT
    text_processed, tokens = tokenizer_text(tokenizer=tokenizer,
                                    text=text)

    print("RESULT ANALYSED\n")
    ner_results = get_predict(model=model_nlp,
                              tokens=tokens,
                              inputs=text_processed)

    return ner_results

# TEXT TO USE
text = "Acrescento que não há de se falar em violação do artigo 114, § 3º, da Constituição Federal, posto que referido dispositivo revela-se impertinente, tratando da possibilidade de ajuizamento de dissídio coletivo pelo Ministério Público do Trabalho, juiz Emerson Rafael, nos casos de greve em atividade essencial. Precisamos falar com a Marcela Santos Moreira em 02/05/2022."

# CALL MODEL
result_model = orchestra_predict(text=text)