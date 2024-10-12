from flask import Flask, render_template, request
from markupsafe import escape

app = Flask(__name__)

profiles = {"admin": "1234"}

@app.route("/")
def homepage():
    return render_template("proj_index.html")

@app.get("/login")
def login_get():
    return render_template("proj_login_form.html")

@app.post("/login")
def login_post():
    # 0: Account doesn't exist, 1: Wrong Password, 2: Blank field(s)
    username = request.form['username']
    password = request.form['password']
    if (not username.isspace() and not password.isspace()):
        # Valid info given
        if (username in profiles):
            # A profile with the given username exists
            if (profiles[username] == password):
                # The password of the given username matches
                return render_template("proj_profile_page.html.jinja", username=username)
            else:
                # The password of the given username is incorrect
                return render_template("proj_login_error.html.jinja", error=1, username=username)
        else:
            # A profile with the given username does not exist
            return render_template("proj_login_error.html.jinja", error=0, username=username)
    else:
        # One or both was/were whitespace
        return render_template("proj_login_error.html.jinja", error=2)
