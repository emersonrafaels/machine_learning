from transformers import BertForTokenClassification, DistilBertTokenizerFast, pipeline


def init_model():

    model = BertForTokenClassification.from_pretrained('monilouise/ner_pt_br')
    tokenizer = DistilBertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                        model_max_length=512,
                                                        do_lower_case=True
                                                        )

    nlp = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)

    return nlp


def get_predict(model, text):

    result = model(text)

    return result


def orchestra_predict(text_to_predict):

    model_nlp = init_model()

    print("RESULT ANALYSED\n")
    ner_results = get_predict(model=model_nlp, text=text_to_predict)

    for result in ner_results:
        print(result)

    return ner_results

# TEXT TO USE
text = "Acrescento que não há de se falar em violação do artigo 114, § 3º, da Constituição Federal, posto que referido dispositivo revela-se impertinente, tratando da possibilidade de ajuizamento de dissídio coletivo pelo Ministério Público do Trabalho, juiz Emerson Rafael, nos casos de greve em atividade essencial. Precisamos falar com a Marcela Santos Moreira em 02/05/2022."

# CALL MODEL
result_model = orchestra_predict(text_to_predict=text)