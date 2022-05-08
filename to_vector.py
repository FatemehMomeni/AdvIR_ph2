from elasticsearch import Elasticsearch
from fastText.python.fasttext_module import fasttext

ELASTIC_PASSWORD = "elasticsearch password:)"
es = Elasticsearch('https://localhost:9200', verify_certs=False, basic_auth=("elastic", ELASTIC_PASSWORD))

for i in range(5440):
    resp = es.get(index='ted_index', id=i)
    with open('ted_index.doc'+str(i)+'.txt', 'w', encoding='utf-8') as f:
        f.write(resp['_source']['transcript'])
    model = fasttext.train_unsupervised('ted_index.doc'+str(i)+'.txt')
    for j in model.words:
        with open('vector.doc'+str(i)+'.txt', 'w', encoding='utf-8') as file:
            file.write(str(model.get_word_vector(j)))
        file.close()
    f.close()
