from SPARQLWrapper import SPARQLWrapper, JSON

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

        # processed.append(f"{abstract} {genre_triple} {directors_triple}")


    return processed


try:
    results = sparql.query().convert()
    # results = {
    #     "head": {"vars": ["title", "abstract", "directors_str", "genres_str"]},
    #     "results": {"bindings": [
    #         {
    #             "title": {"type": "literal", "xml:lang": "en", "value": "Mula sa Puso"},
    #             "abstract": {"type": "literal", "xml:lang": "en",
    #                          "value": "Mula sa Puso (English: From the Heart) is a 1997 Philippine primetime melodrama romance television series originally aired by ABS-CBN from March 10, 1997, to April 9, 1999 (the series ended for 2 years in its long-run) replacing Maria Mercedes and was replaced by Saan Ka Man Naroroon. Claudine Barretto, Rico Yan, and Diether Ocampo played the roles of the main protagonists in the series. It was re-aired in 2008 through Studio 23 and Kapamilya Channel, which are both ABS-CBN subsidiaries. A 2011 remake, starring Lauren Young, JM de Guzman and Enrique Gil, aired on ABS-CBN from March 28, 2011, to August 12, 2011. The show also gave critical acclaim to director Wenn V. Deramas as his first prime time soap project and as a director and gave character actress Princess Punzalan critical acclaim for her character as Selina Pereira-Matias, the main antagonist of the series. Mula sa Puso is known to be the first middle-class-themed Filipino primetime TV drama.Its highest viewed on TV is the bus explosion scene, where Selina killed Via by bombing the bus. It can also seen on Episode 366 on Youtube.It was also known for being the most competitive TV series in the country's TV ratings by mid-1998, alongside Esperanza, which also ran from 1997 to 1999; the popularity of the two teleseryes spawned two films both released in 1999 (Mula sa Puso in February and Esperanza in December), and a joint soundtrack entitled \"Mula Sa Puso ni Esperanza\". The TV series had various crossovers with prime time dramas such as Esperanza and the short-lived miniseries Sa Sandaling Kailangan Mo Ako. The show was aired from Mondays to Fridays at 6:30 pm (March 10, 1997, until January 1, 1999) and later at 8 pm (January 4, 1999, until April 9, 1999) after TV Patrol. The story depicts on the life of Via and as she gracefully turns 18, she will discover she lives a life full of deceit. She will also discover the truth about the identities of her loved ones and the unstoppable troubles when it comes to love and life itself as she falls in love with her savior Gabriel while she also falls into a love triangle with her persistent longtime childhood friend Michael. The series is currently streaming on Jeepney TV's YouTube Channel everyday at 3:00 pm & 3:30 pm, right after Recuerdo de Amor (which also starred Diether Ocampo). The bus explosion of Mula sa Puso is also iconic. When Selina (Princess Punzalan) killed Via (Claudine Barreto) by sending her henchmen in the bus where Via ride. The Bus Explosion - Episode 366 - 367."},
    #             "directors_str": {"type": "literal", "value": "Khryss Adalia, Wenn Deramas"},
    #             "genres_str": {"type": "literal", "value": "Romance film, Drama film"}
    #         },
    #     ]}
    # }

    print("Query executed successfully. Processing results...")
    model_input = generate_model_input(results)

    for t in model_input:
        print(t)

except Exception as e:
    print(f"An error occurred during SPARQL query execution: {e}")
    results = None # Ensure results is None if an error occurred
