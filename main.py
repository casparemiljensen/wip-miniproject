from SPARQLWrapper import SPARQLWrapper, JSON
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


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


def generate_model_input(data):
    processed = []
    entities = data["head"]
    names = entities["vars"]
    for t in data["results"].get("bindings"):
        title = t[names[0]]["value"]
        abstract = t[names[1]]["value"]
        directors = t[names[2]]["value"]
        genres = t[names[3]]["value"]
        genre = genres.split(', ')[0].strip()
        director = directors.split(', ')[0].strip()
        genre_triple = f"{title} → genre → {genre} . "
        director_triple = f"{title} → director → {director} . "
        processed.append({
            "input": f"{abstract} {genre_triple} {director_triple}",
            "label": genre,
        })
    return processed


def execute_query():
    """Run the SPARQL query and return the results object."""
    try:
        print("Executing SPARQL query... This might take a moment.")
        return sparql.query().convert()
    except Exception as e:
        print(f"An error occurred during SPARQL query execution: {e}")
        return None


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def prepare_dataloader(data, tokenizer, batch_size=8):
    texts = [d["input"] for d in data]
    label_names = [d["label"] for d in data]
    label_set = sorted(set(label_names))
    label_map = {label: i for i, label in enumerate(label_set)}
    labels = [label_map[l] for l in label_names]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    dataset = TextDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, len(label_set)


def create_model(num_labels):
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)


def train_model(model, dataloader, epochs=1, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


def main():
    results = execute_query()
    if not results:
        return
    model_input = generate_model_input(results)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataloader, num_labels = prepare_dataloader(model_input, tokenizer)
    model = create_model(num_labels)
    train_model(model, dataloader, epochs=1)
    for entry in model_input:
        print(entry)


if __name__ == "__main__":
    main()
