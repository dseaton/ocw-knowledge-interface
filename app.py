from flask import Flask, render_template, request, jsonify

from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phraser
import networkx as nx
import pandas as pd

import input_terms

app = Flask(__name__)

model = Doc2Vec.load("data/d2v.model")
bigram_transformer = Phraser.load("data/bigram_transformer_model.pkl")
lookup_df = pd.read_csv("data/lookup_df.csv")

find_by_examples = [input_terms.gensim_custom_preprocess(t) for t in input_terms.find_by_examples]
find_by_examples = [bigram_transformer[e][0] for e in find_by_examples]

find_by_topic_second_level = [bigram_transformer[input_terms.gensim_custom_preprocess(t)][0] for t in input_terms.find_by_topic_second_level]

def _get_similar_video_clips(input_term, topn=5):
    sim_docs = model.docvecs.most_similar([model.wv.get_vector(input_term)], topn=topn)

    similar_video_clips = []
    for i, score in sim_docs:
        lookup_df.iloc[i]
        similar_video_clips.append([
            lookup_df.iloc[i]['youtube_id'],
            lookup_df.iloc[i]['course_num'],
            lookup_df.iloc[i]['course_title'],
            lookup_df.iloc[i]['start'],
            lookup_df.iloc[i]['url'],
        ])
    return similar_video_clips

def _get_similar_words(input_term, topn=12):
    return [w for w, score in model.wv.most_similar(input_term, topn=topn)]

@app.route("/network_data/<input_term>")
def network_data(input_term="linear_algebra", topn=10, first_neighbors=5):
    G = nx.Graph()
    sims = model.wv.most_similar(input_term, topn=topn)

    G.add_node(input_term, group="1")
    for w,s in sims:
        G.add_node(w, group="2")
        G.add_edge(input_term, w, weight=s) # model.wv.wmdistance(input_term, w))
    
    for f in range(0, first_neighbors):
        for w,s in model.wv.most_similar(sims[f][0], topn=topn):
            if G.has_node(w) == False: 
                G.add_node(w, group=str(f+3))
            G.add_edge(sims[f][0], w, weight=s) # model.wv.wmdistance(sims[0][0], w))
    
    # pos = nx.spring_layout(G)

    return jsonify(nx.json_graph.node_link_data(G))

@app.route('/network', defaults={'input_term': 'einstein'})
@app.route("/network/<input_term>")
def network(input_term="einstein"):
    return render_template(
        "network.html",
        input_term = input_term,
    )

@app.route('/topics', defaults={'input_term': 'deep_learning'})
@app.route("/topics/<input_term>")
def topics(input_term="deep_learning"):
    return render_template(
        "topics.html", 
        input_term = input_term,
        topics = find_by_topic_second_level, 
        similar_words = _get_similar_words(input_term),
        similar_video_clips = _get_similar_video_clips(input_term)
    )

@app.route('/words', defaults={'input_term': 'deep_learning'})
@app.route("/words/<input_term>")
def words(input_term="deep_learning"):
    return render_template(
        "words.html", 
        input_term = input_term,
        examples = find_by_examples, 
        similar_words = _get_similar_words(input_term),
        similar_video_clips = _get_similar_video_clips(input_term)
    )

@app.route("/")
def home():
    return render_template(
        "index.html",
        examples = find_by_examples,
    )

@app.route("/about")
def about():
    return render_template(
        "about.html",
    )


if __name__ == '__main__':
    app.run(debug=True)
