import ensemble 
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

f = open('models.json')

Models = json.load(f)


class Mental(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    post = db.Column(db.String(200), nullable=False)
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
        # modelUsed = request.form['models']
        x = [task_content]
        y=ensemble.predict(x)
        print(y)
        new_post = Mental(post=task_content, classification=y)
        try:
            db.session.add(new_post)
            db.session.commit()
            return redirect('/')
        except Exception:
            return "Error"
        # return task_content
    else:
        tasks = Mental.query.order_by(Mental.date_created).all()
        return render_template('index.html', tasks=tasks)


# @app.route('/models')
# def showModels():
#     return render_template('models.html', Models=Models)


@app.route('/delete/<int:id>')
def delete(id):
    task_delete = Mental.query.get_or_404(id)
    try:
        db.session.delete(task_delete)
        db.session.commit()
        return redirect('/')
    except Exception:
        return Exception


# @app.route('/models/model')
# def bert():
#     return render_template("bert.html", desc=Models['bert']['desc'])

if (__name__ == "__main__"):
    app.run(debug=True)
