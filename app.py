import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

# load model
model = pickle.load(open('model.pickle','rb'))

# app
app = Flask(__name__)

#check
@app.route('/')
def home():
    return render_template('wb.html')

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data    
    if request.method == 'POST':
      data = request.form
    # convert data into dataframe
    data_df = pd.DataFrame.from_dict([data])
    #data_df = data_df.astype(np.int64)
    # predictions
    result = model.predict(data_df)
    # send back to browser
    output = {'Predicted Result': int(result[0])}
    # return data
    output = jsonify(output)
    return render_template('wb.html' ,output = output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

