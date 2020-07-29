from flask import Flask, render_template

from gensim.models.doc2vec import Doc2Vec
import pandas as pd

app = Flask(__name__)

model = Doc2Vec.load("data/d2v.model")
viz_df = pd.read_csv("data/model_viz_df.csv")
find_by_examples = [
    "deep_learning", "maxwell", "computation", 
    "calculus", "astrophysics", "electromagnetism", 
    "relativity", "thermodynamics", "gluon",
    "economics", 
]

def _get_similar_video_clips(input_term, topn=5):
    sim_docs = model.docvecs.most_similar([model.wv.get_vector(input_term)], topn=topn)

    similar_video_clips = []
    for i, score in sim_docs:
        viz_df.iloc[i]
        similar_video_clips.append([
            viz_df.iloc[i]['youtube_id'],
            viz_df.iloc[i]['course_id'],
            int(viz_df.iloc[i]['id']/1000 * 360)
        ])
    return similar_video_clips

def _get_similar_words(input_term, topn=10):
    return [w for w, score in model.wv.most_similar(input_term, topn=topn)]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/words/<input_term>")
def words(input_term="deep_learning"):
    return render_template(
        "words.html", 
        input_term = input_term,
        examples = find_by_examples, 
        similar_words = _get_similar_words(input_term),
        similar_video_clips = _get_similar_video_clips(input_term)
    )


if __name__ == '__main__':
    app.run(debug=True)
