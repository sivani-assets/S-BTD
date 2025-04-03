import os
import tensorflow as tf
import keras
import numpy as np
import cv2
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors


from datetime import datetime
import os



app = Flask(__name__)
app.secret_key = 'your_secret_key'
DB_NAME = 'database.db'

UPLOAD_FOLDER = 'static/uploads'

REPORT_FOLDER = 'static/reports'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load the model
model = keras.models.load_model('BrainTumor10Epochs.keras')
print('Model loaded. Check http://127.0.0.1:5000/')


# Run dummy prediction to initialize model input/output
model.predict(np.zeros((1, 64, 64, 3)))


def get_className(classNo):
    """Convert class index to human-readable label"""
    return "No Brain Tumor" if classNo == 0 else "Yes Brain Tumor"


def getResult(img_path):
    """Process image and predict class"""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))  # Resize image to match model input
    image = np.array(image) / 255.0  # Normalize the image
    input_img = np.expand_dims(image, axis=0)

    prediction = model.predict(input_img)
    result = np.argmax(prediction, axis=-1)  # Get the class index
    return result[0]

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Login users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        """)
        # Patient reports table (separate)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT,
                name TEXT,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                result TEXT,
                date TEXT,
                report_path TEXT
            )
        """)
        conn.commit()


# Routes for different pages


@app.route('/')
def home():
    return render_template('home.html')  # Home page


@app.route('/cnn')
def cnn():
    return render_template('cnn.html')


@app.route('/FAQ')
def FAQ():
    return render_template('FAQ.html')

@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                return "Username already exists!"
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username = ?", (username,))
            user = cursor.fetchone()
            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                return redirect(url_for('home'))
            else:
                return "Invalid credentials!"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT scan_id, name, result, date, report_path FROM patient_reports ORDER BY id DESC")
        reports = cursor.fetchall()
    return render_template('dashboard.html', reports=reports)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            class_index = np.argmax(model.predict(np.expand_dims(cv2.resize(cv2.imread(file_path), (64, 64)) / 255.0, axis=0)))
            result = get_className(class_index)

            return render_template('report_form.html', filename=filename, result=result)

    return render_template('predict.html')


@app.route('/report_form')
def report_form():
    return render_template('report_form.html')


@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    scan_id = request.form['scan_id']
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    contact = request.form['contact']
    result = request.form.get('result')  # Ensure it's fetched from form

    filename = request.form['filename']
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_path = os.path.join(REPORT_FOLDER, f"{scan_id}_{name}_report.pdf")

    # Generate PDF
    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    pale_blue = colors.HexColor("#e6f0ff")
    c.setFillColor(pale_blue)
    c.rect(0, 0, width, height, fill=True, stroke=0)

    c.setStrokeColor(colors.black)
    c.setLineWidth(2)
    c.rect(30, 30, width - 60, height - 60)


# --- PDF Header Branding (S in white circle + BTD in blue) ---
    c.setFillColor(colors.white)
    c.circle(75, height - 75, 25, fill=1, stroke=0)  # White Circle

    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(colors.HexColor("#458ff6"))
    c.drawString(67, height - 83, "S")

    c.setFont("Helvetica-Bold", 22)
    c.drawString(110, height - 83, "BTD")

    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.black)
    c.drawString(400, height - 60, f"Date: {date}")

    # Header Text (below logo)
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.blue)
    c.drawString(180, height - 110, "Brain Tumor Detection Report")


    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)

    table_data = [
        ["Scan ID", scan_id],
        ["Patient Name", name],
        ["Age", age],
        ["Gender", gender],
        ["Contact", contact],
    ]
    y = height - 160
    for row in table_data:
        c.drawString(60, y, f"{row[0]}: {row[1]}")
        y -= 20

    c.setFillColor(colors.HexColor("#d32f2f") if result == "Yes Brain Tumor" else colors.HexColor("#388e3c"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(150, height - 320, f"Prediction Result: {result}")

    
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        c.drawImage(img_path, 200, y - 320, width=200, height=200, preserveAspectRatio=True)
        c.setFont("Helvetica-Oblique", 10)
        c.setFillColor(colors.darkgray)
        c.drawString(200, y - 330, f"MRI Scan Preview")
    except:
        pass


    diagnosis = "No abnormal mass. No signs of neoplasm." if result == "No Brain Tumor" else "Presence of tumor detected. Consult a neurologist."
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 650, f"MRI Scan Diagnosis: {diagnosis}")


    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    c.drawString(60, 50, "Generated by S BTD Diagnostic System")



    
    c.save()

    # Insert into database
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patient_reports (scan_id, name, age, gender, contact, result, date, report_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (scan_id, name, age, gender, contact, result, date, report_path))
        conn.commit()

    return render_template('report_success.html', name=name, result=result, report_path=report_path)


@app.route('/download_report/<path:path>')
def download_report(path):
    return send_file(path, as_attachment=True)



if __name__ == '__main__':
    init_db()
    app.run(debug=True)
