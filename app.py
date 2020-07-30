from flask import Flask, render_template

from gensim.models.doc2vec import Doc2Vec
import pandas as pd

app = Flask(__name__)

model = Doc2Vec.load("data/d2v.model")
lookup_df = pd.read_csv("data/lookup_df.csv")
find_by_examples = [
    "deep_learning", "computation", "artificial_intelligence",
    "astrophysics", "electromagnetism", "maxwell", 
    "relativity", "thermodynamics", "gluon",
    "statistics", "statistical_mechanics",
    "programming", "electronics", "signal_processing", "nanotechnology", 
    "architecture", "music",
    "medical_imaging", "literature", "fiction",
    "linear_algebra", "topology", "calculus", "probability", 
    "biophysics", "organic_chemistry", "anthropology",
    "econometrics", "economics", "project_management", "innovation",
    # "social_justice", # "urban_planning", #"educational_technology",
]

def _get_similar_video_clips(input_term, topn=5):
    sim_docs = model.docvecs.most_similar([model.wv.get_vector(input_term)], topn=topn)

    similar_video_clips = []
    for i, score in sim_docs:
        lookup_df.iloc[i]
        similar_video_clips.append([
            lookup_df.iloc[i]['youtube_id'],
            lookup_df.iloc[i]['course_num'],
            lookup_df.iloc[i]['course_title'],
            lookup_df.iloc[i]['start']
        ])
    return similar_video_clips

def _get_similar_words(input_term, topn=12):
    return [w for w, score in model.wv.most_similar(input_term, topn=topn)]

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
