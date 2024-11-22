from flask import Flask, request, render_template, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from heapq import nlargest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
import os
import random
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    action = db.Column(db.Float, default=0.0)
    adventure = db.Column(db.Float, default=0.0)
    animation = db.Column(db.Float, default=0.0)
    comedy = db.Column(db.Float, default=0.0)
    crime = db.Column(db.Float, default=0.0)
    documentary = db.Column(db.Float, default=0.0)
    drama = db.Column(db.Float, default=0.0)
    family = db.Column(db.Float, default=0.0)
    fantasy = db.Column(db.Float, default=0.0)
    history = db.Column(db.Float, default=0.0)
    horror = db.Column(db.Float, default=0.0)
    music = db.Column(db.Float, default=0.0)
    mystery = db.Column(db.Float, default=0.0)
    romance = db.Column(db.Float, default=0.0)
    science_fiction = db.Column(db.Float, default=0.0)
    tv_movie = db.Column(db.Float, default=0.0)
    thriller = db.Column(db.Float, default=0.0)
    war = db.Column(db.Float, default=0.0)
    western = db.Column(db.Float, default=0.0)
    watched_shows = db.Column(db.Text, default='[]')  

    def __init__(self, username, password, genres):
        self.username = username
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        genre_value = 1.0 / len(genres)
        for genre in genres:
            setattr(self, genre.lower().replace(' ', '_'), genre_value)

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        genres = request.form.getlist('genres')

        new_user = User(username=username, password=password, genres=genres)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['username'] = user.username
            session['watched_shows'] = json.loads(user.watched_shows)
            print("Login successful, session variables set")
            return redirect('/dashboard')
        else:
            flash('Incorrect username or password', 'danger')
            return render_template('login.html')

    return render_template('login.html')

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 350)
        self.fc2 = nn.Linear(350, 350)
        self.fc3 = nn.Linear(350, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def context(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        
        return x

model = DNN(input_dim=40)
model_path = "dnn_model.pth"  
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

def generate_user_context(user):
    return [
        user.action,
        user.adventure,
        user.animation,
        user.comedy,
        user.crime,
        user.documentary,
        user.drama,
        user.family,
        user.fantasy,
        user.history,
        user.horror,
        user.music,
        user.mystery,
        user.romance,
        user.science_fiction,
        user.tv_movie,
        user.thriller,
        user.war,
        user.western
    ]

def extract_context(dnn_model, X_tensor):
    with torch.no_grad():
        features = dnn_model.context(X_tensor).numpy()
    return features

def choose_arm(contexts, A, b, alpha):
    num_arms = len(contexts)
    max_pta = -np.inf
    chosen_arm = None

    for a in range(num_arms):
        x_a = contexts[a].reshape(-1, 1)
        A_inv = np.linalg.inv(A[a])
        theta_a = np.dot(A_inv, b[a])
        pta = np.dot(theta_a.T, x_a) + alpha * np.sqrt(np.dot(x_a.T, np.dot(A_inv, x_a)))

        if pta > max_pta:
            max_pta = pta
            chosen_arm = a

    return chosen_arm, max_pta

def choose_top_arms(contexts,var, A, b, alpha, user_preferences, df, top_n=6):
    num_arms = len(contexts)
    pta_values = []
    watched_shows = session.get('watched_shows', [])  

    for a in range(num_arms):
        var = var + 1
        if a in watched_shows:
            continue
            
        
        item_genres = str(df.iloc[a]['Genre']).split(', ')
        valid_item = False
        for genre in item_genres:
            genre_pref = user_preferences.get(genre, 0)
            if genre_pref > 0:
                valid_item = True
                break
                
        if valid_item:
            x_a = contexts[a].reshape(-1, 1)
            A_inv = np.linalg.inv(A[a])
            theta_a = np.dot(A_inv, b[a])
            pta = np.dot(theta_a.T, x_a) + alpha * np.sqrt(np.dot(x_a.T, np.dot(A_inv, x_a)))
            pta_values.append((pta, a))
    
    
    if len(pta_values) < top_n:
        for a in range(num_arms):
            if a not in watched_shows and not any(a == pair[1] for pair in pta_values):
                x_a = contexts[a].reshape(-1, 1)
                A_inv = np.linalg.inv(A[a])
                theta_a = np.dot(A_inv, b[a])
                pta = np.dot(theta_a.T, x_a) + alpha * np.sqrt(np.dot(x_a.T, np.dot(A_inv, x_a)))
                pta_values.append((pta, a))
    
    top_arms = []
    for _ in range(min(top_n, len(pta_values))):
        max_pta = max(pta_values, key=lambda x: x[0])
        top_arms.append(max_pta[1])
        pta_values.remove(max_pta)

    return var,top_arms, pta_values

A_file_path = 'A.npy'
b_file_path = 'b.npy'
df = pd.read_csv('dataset.csv', lineterminator='\n')

def load_A_b():
    if os.path.exists(A_file_path) and os.path.exists(b_file_path):
        A = np.load(A_file_path, allow_pickle=True)
        b = np.load(b_file_path, allow_pickle=True)
    else:
        A = [np.identity(350) for _ in range(len(df))]
        b = [np.zeros((350, 1)) for _ in range(len(df))]
    return A, b

def save_A_b(A, b):
    np.save(A_file_path, A)
    np.save(b_file_path, b)

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return redirect('/login')
            
        # Get watched shows
        watched_shows = json.loads(user.watched_shows)
        # Initialize empty watch_again list
        watch_again = []
        
        if watched_shows:
            # Convert to set to remove any duplicates
            watched_shows = list(set(watched_shows))
            # Get min between 6 and total watched shows
            num_shows = min(6, len(watched_shows))
            if num_shows > 0:
                # Randomly sample without replacement
                random_indices = random.sample(watched_shows, num_shows)
                watch_again = [(df.iloc[idx, -1], idx) for idx in random_indices]
        
        genres = {genre: getattr(user, genre.lower().replace(' ', '_')) for genre in [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'
        ] if getattr(user, genre.lower().replace(' ', '_')) > 0}
        
        
        user_data = generate_user_context(user)
        
        item_contexts = df['Genre'].str.get_dummies(sep=', ')

        if item_contexts.shape[1] < 19:
            additional_columns = 19 - item_contexts.shape[1]
            zeros_df = pd.DataFrame(np.zeros((item_contexts.shape[0], additional_columns)), columns=[f'col_{i}' for i in range(additional_columns)])
            item_contexts = pd.concat([item_contexts, zeros_df], axis=1)

        item_contexts = item_contexts.fillna(0)

        vote_average = df['Vote_Average'].values.reshape(-1, 1)
        popularity = df['Popularity'].values.reshape(-1, 1)

        item_data = np.hstack((item_contexts.values, vote_average, popularity))
        
        A, b = load_A_b()

        contexts = []
        for a in range(len(df)):
            current_context = user_data + item_data[a].tolist()
            X_tensor = torch.tensor(current_context, dtype=torch.float32)
            x_ta = extract_context(model, X_tensor.unsqueeze(0))
            contexts.append(x_ta)

        print("Contexts generated")

        
        cnt,top_arms, pta_values = choose_top_arms(contexts,0, A, b, 1.0, genres, df, top_n=6)
        ind = random.randint(0, len(top_arms) - 1)
        i = random.randint(0,99)
        if i not in top_arms:
            top_arms[ind] = i   
        print("Top arms:", top_arms)
        print(cnt)
        top_image_urls = [(df.iloc[arm, -1], df.index[arm]) for arm in top_arms]

        session['selected_arms'] = top_arms
        session['selected_arms_contexts'] = [contexts[arm].tolist() for arm in top_arms]
        
        
        save_A_b(A, b)
        
        return render_template('dashboard.html', 
                            user=user, 
                            genres=genres, 
                            top_image_urls=top_image_urls,
                            watch_again=watch_again,
                            df=df)
    
    print("User not in session")
    return redirect('/login')


@app.route('/remove_show', methods=['POST'])
def remove_show():
    if 'username' in session:
        show_id = int(request.form['show_id'])
        user = User.query.filter_by(username=session['username']).first()
        watched_shows = json.loads(user.watched_shows)
        watched_shows.append(show_id)
        user.watched_shows = json.dumps(watched_shows)
        db.session.commit()
        session['watched_shows'] = watched_shows

        A, b = load_A_b()
        selected_arms = session.get('selected_arms', [])
        selected_arms_contexts = session.get('selected_arms_contexts', [])

        try:
            
            arm_index = selected_arms.index(show_id)
            
            if arm_index < len(selected_arms_contexts):
                context = selected_arms_contexts[arm_index]
                if context:
                    x_a = np.array(context).reshape(-1, 1)
                    A[show_id] += np.dot(x_a, x_a.T)
                    b[show_id] += 1 * x_a  
                    save_A_b(A, b)
        except (ValueError, IndexError) as e:
            print(f"Error updating matrices: {e}")
            

        
        session.pop('selected_arms', None)
        session.pop('selected_arms_contexts', None)

        return redirect('/dashboard')
    return redirect('/login')

@app.route('/not_interested', methods=['POST'])
def not_interested():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return redirect('/login')

        A, b = load_A_b()
        top_arms = session.get('selected_arms', [])
        selected_arms_contexts = session.get('selected_arms_contexts', [])

        for i, arm in enumerate(top_arms):
            if arm < len(A):
                context = np.array(selected_arms_contexts[i]).reshape(-1, 1)
                A[arm] += np.dot(context, context.T)

        save_A_b(A, b)

        session.pop('selected_arms', None)
        session.pop('selected_arms_contexts', None)

        return redirect('/dashboard')
    return redirect('/login')

@app.route('/logout')
def logout():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
