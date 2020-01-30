import pickle
import numpy as numpy
from fastai import *
from fastai.text import *
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource


app = Flask(__name__)
api = Api(app)

with open("../list_of_sentences", "r+") as f:
    documents = f.readlines()
for i in range(len(documents)):
    documents[i] = documents[i][:-1]

path = untar_data(
    URLs.IMDB
)  # this will download a 176 MB tgz file(only downloads once)

data_lm = (
    TextList.from_folder(path)
    .filter_by_folder(include=["train", "test", "unsup"])
    .split_by_rand_pct(0.01, seed=42)
    .label_for_lm()
    .databunch(bs=32, num_workers=-1)
)

learn = language_model_learner(data_lm, AWD_LSTM)


def get_one_item(learn, doc):
    xb, yb = learn.data.one_item(doc)
    return xb


def encode_doc(learn, doc):
    xb = get_one_item(learn, doc)
    lstm_encoder = learn.model[0]
    lstm_encoder.reset()
    with torch.no_grad():
        out = lstm_encoder.eval()(xb)
    return out[0][2][0][-1].detach().numpy()


document_matrix = []
for doc in documents:
    doc_vector = encode_doc(learn, doc)
    document_matrix.append(doc_vector)

document_matrix = np.array(document_matrix)


def find_similar(documents, document_matrix):
    """
    find the similar documents based on the cosine distance of two vectors
    """
    similar_list = []
    for i in range(len(documents)):
        sim = cosine_similarity(document_matrix[i : i + 1], document_matrix)[0]
        #         print((sim))
        indexes = [i for i, s in enumerate(sim) if np.logical_and(s < 0.99, s > 0.0)]
        idx_sim_pairs = {}
        for idx in indexes:
            idx_sim_pairs[int(idx)] = sim[idx]
        idx_sim_pairs = {
            k: v
            for k, v in sorted(
                idx_sim_pairs.items(), key=lambda item: item[1], reverse=True
            )
        }
        #         print(idx_sim_pairs)
        sim_sentences = [documents[i]]
        sim_sentences.extend([documents[i] for i in list(idx_sim_pairs.keys())[:2]])
        similar_list.append(sim_sentences)

    return similar_list


class smililar_docs(Resource):
    def get(self):
        similar_out = find_similar(documents, document_matrix)
        output = {"grouped_similar": similar_out}
        return output


api.add_resource(smililar_docs, "/")

if __name__ == "__main__":
    # print(find_similar(documents, document_matrix))
    app.run(debug=True)
