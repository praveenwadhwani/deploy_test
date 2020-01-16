#import pandas as pd
from flask import Flask, request, render_template, Response
#import pickle

# app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug = True)
