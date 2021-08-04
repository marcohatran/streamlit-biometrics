from elasticsearch import Elasticsearch


def face_search(face_vector):
    index = "faces_demo"
    es = Elasticsearch()
    response = es.search(
        index=str(index),
        body={
            "size": 1,
            # "_source": "cif",
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'face_encoding')",
                        "params": {
                            "query_vector": face_vector
                        }
                    }
                }
            }
        }
    )
    results = []
    for hit in response['hits']['hits']:
        if float(hit['_score']) > 0.9:
            cif = hit.get('_source').get('cif') if hit.get('_source').get('cif') else None
            phone = hit.get('_source').get('phone') if hit.get('_source').get('phone') else None
            score = hit.get('_score')
            results.append({"score": score,
                            "cif": cif,
                            "phone": phone})

    return results
