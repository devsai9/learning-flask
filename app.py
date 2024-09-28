from flask import Flask, render_template, request
from markupsafe import escape

app = Flask(__name__)

# Simple Routing 
@app.route("/")
def homepage():
    return "<h1>Hello World!</h1>"

# Rendering Templates
@app.route("/html-file")
def render_index_html_file():
    return render_template("test_template.html")

# Using URL Parameters
@app.route("/name-input/<name>")
def name_input(name):
    return f"Hello, {escape(name)}!"

@app.route("/number-input/<int:num>")
def num_input(num):
    return f"You have provided the number {num} :o"