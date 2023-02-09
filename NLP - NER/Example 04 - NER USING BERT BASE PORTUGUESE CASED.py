import torch
from transformers import AutoTokenizer, AutoModel

def init_model():

    # Load the pre-trained NER model
    model_name = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return model, tokenizer

def tokenizer_text(tokenizer, text):

    # Encode the text for processing by the model
    inputs = tokenizer.encode(text, return_tensors="pt")

    return inputs

def get_predict(model, tokenizer, inputs):

    # Generate predictions from the model
    with torch.no_grad():
        outputs = model(inputs)
        _, prediction = torch.max(outputs[0], dim=2)

    # Extract the named entities
    named_entities = []
    entity = ""
    previous_tag = "O"

    for word, tag in zip(tokenizer.convert_ids_to_tokens(inputs[0].tolist()), prediction[0].tolist()):
        print("WORD: {} - TAG: {}".format(word, tag))

        """
        tag = idx2tag[tag]
        if tag == "O":
            if entity:
                named_entities.append((entity, previous_tag[2:]))
                entity = ""
        elif tag[0] == "B":
            if entity:
                named_entities.append((entity, previous_tag[2:]))
            entity = word
        else:
            entity += " " + word
        previous_tag = tag

    # Print the named entities
    for named_entity in named_entities:
        print(named_entity)
    """

    return named_entities

def orchestra_predict(text):

    # INIT MODEL
    model_nlp, tokenizer = init_model()

    # PREPROCESSING TEXT
    text_processed = tokenizer_text(tokenizer=tokenizer,
                                    text=text)

    print("RESULT ANALYSED\n")
    ner_results = get_predict(model=model_nlp,
                              tokenizer=tokenizer,
                              inputs=text_processed)

    for result in ner_results:
        print(result)

    return ner_results

# TEXT TO USE
text = "Acrescento que não há de se falar em violação do artigo 114, § 3º, da Constituição Federal, posto que referido dispositivo revela-se impertinente, tratando da possibilidade de ajuizamento de dissídio coletivo pelo Ministério Público do Trabalho, juiz Emerson Rafael, nos casos de greve em atividade essencial. Precisamos falar com a Marcela Santos Moreira em 02/05/2022."

# CALL MODEL
result_model = orchestra_predict(text=text)
