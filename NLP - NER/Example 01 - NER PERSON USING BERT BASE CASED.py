import tensorflow as tf
import transformers

model = transformers.TFBertForTokenClassification.from_pretrained("bert-base-cased")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

text = "VALIDO EM TODO O TERRITÓRIO NACIONAL 13/12/2010 EMERSONV RAFAEL FILIAÇÃO MARCELA DOS SANTOS EDVALDO CARLOS"

input_ids = tokenizer.encode(text, return_tensors='tf')

"""

with tf.device("/cpu:0"):
    output = model(input_ids).last_hidden_state

# NER Labels
labels = ["O", "B-PER", "I-PER", "X", "[CLS]", "[SEP]"]

predictions = tf.argmax(output, axis=2)

names = []
name = ""
for word, pred in zip(text.split(), predictions[0].numpy()):
    if labels[pred] == "B-PER":
        if name:
            names.append(name)
            name = ""
        name += word
    elif labels[pred] == "I-PER":
        name += " " + word
    elif labels[pred] == "O" and name:
        names.append(name)
        name = ""

print("Names:", names)

"""