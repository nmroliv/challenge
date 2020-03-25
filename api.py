import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
from pandas.io.json import json_normalize
from flask import Flask, request, jsonify, render_template, abort

# to load the saved model
xgb_model = pickle.load(open("challenge.pkl", "rb"))

app = Flask(__name__)


@app.route('/')
def home():
    return "Challenge Machine Learning"


@app.route('/results',methods=['POST'])
def results():
    tab = request.get_json(force=True)
    tab = pd.read_json(tab,orient="records")
    try:
        tab = tab.astype("int64")
    except:
        abort(400, {'message': 'Feature types do not match'})
    if np.sum(np.isin(xgb_model.feature_names,tab.columns.values)==False)>0:
        abort(400, {'message': 'Feature names do not match'})
    tab = tab[xgb_model.feature_names]
    Dtab = xgb.DMatrix(tab)
    output = xgb_model.predict(Dtab)
    return jsonify(output.tolist())


if __name__ == "__main__":
    app.run(debug=True)


