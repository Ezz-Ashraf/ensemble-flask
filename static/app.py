# import bertConfig as bert
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import json
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

f = open('models.json')

# returns JSON object as
# a dictionary
Models = json.load(f)

# Iterating through the json
# list
print(Models['lstm'])
# prediction function


def LRPredictor(to_predict):
    loaded_model = pickle.load(open("LR_reddit_700.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    disorders = ['Anxiety', 'BPD', 'bipolar',
                 'depression', 'mentalillness', 'schizophrenia']
    return disorders[result[0]]


def NBPredictor(to_predict):
    loaded_model = pickle.load(open("NB_reddit_700.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    disorders = ['Anxiety', 'BPD', 'bipolar',
                 'depression', 'mentalillness', 'schizophrenia']
    return disorders[result[0]]


class Mental(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post = db.Column(db.String(200), nullable=False)
    AIModel = db.Column(db.String(200), nullable=False)
    classification = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id


with app.app_context():
    db.create_all()


@app.route('/', methods=['POST', 'GET'])
def index():
    if (request.method == 'POST'):
        task_content = request.form['content']
        modelUsed = request.form['models']
        x = [task_content]
        print(modelUsed)
        if modelUsed == "LR":
            y = LRPredictor(x)
        elif modelUsed == "NB":
            y = NBPredictor(x)
        # elif modelUsed=="BERT":
        #     y=bert.bertPredict(x)
        print(y)
        # print(bert.bertPredict(["paranoia"]))
        new_post = Mental(post=task_content,
                          AIModel=modelUsed, classification=y)
        try:
            db.session.add(new_post)
            db.session.commit()
            return redirect('/')
        except:
            return 'Error Happened'
        # return task_content
    else:
        tasks = Mental.query.order_by(Mental.date_created).all()
        return render_template('index.html', tasks=tasks, Models=Models)


@app.route('/models')
def showModels():
    return render_template('models.html', Models=Models)


@app.route('/delete/<int:id>')
def delete(id):
    task_delete = Mental.query.get_or_404(id)
    try:
        db.session.delete(task_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'Error Happened'


# @app.route('/models/model')
# def bert():
#     return render_template("bert.html", desc=Models['bert']['desc'])


if (__name__ == "__main__"):
    app.run(debug=True)
