from SPARQLWrapper import SPARQLWrapper, JSON

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
import torch
from collections import Counter
import pickle


sparql = SPARQLWrapper("http://dbpedia.org/sparql")


sparql_query = """
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT 
?titleLabel ?abstract
(GROUP_CONCAT(DISTINCT ?directorLabel; SEPARATOR=", ") AS ?directors)
(GROUP_CONCAT(DISTINCT ?genreLabel; SEPARATOR=", ") AS ?genres)

WHERE {
    ?film a dbo:Film .
    ?film rdfs:label ?titleLabel .
    FILTER (lang(?titleLabel) = "en")

    ?film dbo:abstract ?abstract .
    FILTER (lang(?abstract) = "en")

    ?film dbo:genre ?genre .
    ?genre rdfs:label ?genreLabel .
    FILTER (lang(?genreLabel) = "en")

    ?film dbo:director ?director .
    ?director rdfs:label ?directorLabel .
    FILTER (lang(?directorLabel) = "en")
}
 
LIMIT 50

"""

sparql.setQuery(sparql_query)
sparql.setReturnFormat(JSON)

print("Executing SPARQL query... This might take a moment.")


def encode_labels_and_split_data(processed_data):
    labels = [item["label"] for item in processed_data]
    label_counts = Counter(labels)

    filtered_data = [item for item in processed_data if label_counts[item["label"]] >= 2]

    if len(filtered_data) < 2:
        raise ValueError("Not enough data after filtering low-frequency labels.")

    filtered_labels = [item["label"] for item in filtered_data]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_labels)

    for i, item in enumerate(filtered_data):
        item["label"] = int(encoded_labels[i])

    train_data, val_data = train_test_split(
        filtered_data,
        test_size=0.2,
        stratify=encoded_labels,
        random_state=42
    )

    return train_data, val_data, label_encoder


def prepare_dataset_dict(train_data, val_data, tokenizer):
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    })

    def tokenize_function(example):
        return tokenizer(example["input"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["input"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def load_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
        }

    return compute_metrics


def generate_model_input(data):
    processed = []

    entities = data["head"]

    names = entities["vars"]

    for t in data["results"].get("bindings"):
        dict_entry = {}

        title = t[names[0]].__getitem__("value")
        abstract = t[names[1]].__getitem__("value")
        directors = t[names[2]].__getitem__("value")
        genres = t[names[3]].__getitem__("value")

        # We take only first of genre and director
        genre = genres.split(', ')[0].strip()
        director = directors.split(', ')[0].strip()

        genre_triple = f"{title} → genre → {genre} . "
        directors_triple = f"{title} → director → {director} . "

        # You'll want a list where each item in the list is a dictionary with two keys:

        dict_entry['input'] = f"{abstract} {genre_triple} {directors_triple}"
        dict_entry['label'] = genre

        processed.append(dict_entry)

    return processed


try:
    results = sparql.query().convert()

    # results = {
    #     "head": {"vars": ["title", "abstract", "directors_str", "genres_str"]},
    #     "results": {"bindings": [
    #         {
    #             "title": {"type": "literal", "xml:lang": "en", "value": "Mula sa Puso"},
    #             "abstract": {"type": "literal", "xml:lang": "en",
    #                          "value": "Mula sa Puso (English: From the Heart) is a 1997..."},
    #             "directors_str": {"type": "literal", "value": "Khryss Adalia, Wenn Deramas"},
    #             "genres_str": {"type": "literal", "value": "Romance film, Drama film"}
    #         },
    #     ]}
    # }

    print("Query executed successfully. Processing results...")
    processed_data = generate_model_input(results)

    train_data, val_data, label_encoder = encode_labels_and_split_data(processed_data)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = prepare_dataset_dict(train_data, val_data, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(label_encoder.classes_)
    )

    compute_metrics = load_metrics()

    training_args = TrainingArguments(
        output_dir="./bert_genre_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Finetuning BERT on film genres...")
    trainer.train()
    print("Done training.")

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("Label encoder saved as label_encoder.pkl")

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

except Exception as e:
    print(f"SPARQL or training error: {e}")
    results = None
