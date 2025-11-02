# app.py
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from io import BytesIO
from functools import wraps
import logging

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, jsonify, session, send_from_directory
)
from flask_wtf import FlaskForm
from wtforms import PasswordField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask
import os

import requests
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --- OpenWeather API Key ---
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")


# -----------------------
# Create Flask App
# -----------------------
app = Flask(__name__)

# -----------------------
# Configuration
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "app.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mold_model_final.keras")
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "ChangeThisAdminPass!")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
# Flask config
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY"),
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    DB_PATH=DB_PATH,
    MODEL_PATH=MODEL_PATH,
    ALLOWED_EXT=ALLOWED_EXT,
    ADMIN_PASSWORD=ADMIN_PASSWORD,
    MAX_CONTENT_LENGTH=8*1024*1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,   # True in production with HTTPS
    SESSION_COOKIE_SAMESITE="Lax"
)
app.permanent_session_lifetime = timedelta(hours=2)

# -----------------------
# Database Helper Functions 
# -----------------------

import sqlite3
from datetime import datetime, timedelta, timezone
from werkzeug.security import generate_password_hash, check_password_hash

def get_db_conn():
    """Return a SQLite connection with dict-style row access."""
    conn = sqlite3.connect(app.config["DB_PATH"])
    conn.row_factory = sqlite3.Row
    return conn


# -----------------------
# Schema Setup
# -----------------------
def init_db():
    """Initialize the database and ensure all necessary tables exist."""
    conn = get_db_conn()
    c = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            failed_attempts INTEGER DEFAULT 0,
            locked_until TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    """)

    # Uploads table 
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            shap_path TEXT,
            uploaded_at TEXT,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Feedback table
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            rating INTEGER,
            comment TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


# -----------------------
# User / Auth Functions
# -----------------------
def get_user_by_email(email):
    """Fetch a user record by email."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user


def create_user(email, password_hash, role="user"):
    """Insert a new user with hashed password."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO users (email, password_hash, role, failed_attempts, locked_until, created_at)
        VALUES (?, ?, ?, 0, NULL, ?)
    """, (email, password_hash, role, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def increment_failed_attempts(user_id):
    """Increment failed login attempts and lock account temporarily if needed."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT failed_attempts FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()

    if row:
        failed_attempts = (row["failed_attempts"] or 0) + 1
        locked_until = None
        if failed_attempts >= 5:
            # Lock for 15 minutes after 5 failed attempts
            locked_until = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
            failed_attempts = 0

        c.execute("""
            UPDATE users
            SET failed_attempts = ?, locked_until = ?
            WHERE id = ?
        """, (failed_attempts, locked_until, user_id))

    conn.commit()
    conn.close()


def update_user_login_success(user_id):
    """Reset failed attempts and update last login time."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        UPDATE users
        SET failed_attempts = 0, locked_until = NULL, last_login = ?
        WHERE id = ?
    """, (datetime.now(timezone.utc).isoformat(), user_id))
    conn.commit()
    conn.close()


# -----------------------
# Uploads and Feedback
# -----------------------
def add_record(filename, pred_label, confidence, shap_path=None, user_id=None):
    """Insert a new prediction record into uploads table."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO uploads (filename, prediction, confidence, shap_path, uploaded_at, user_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, pred_label, confidence, shap_path, now, user_id))
    conn.commit()
    conn.close()


def list_records(limit=None):
    """Return all records from uploads table, optionally limited."""
    conn = get_db_conn()
    c = conn.cursor()
    query = "SELECT * FROM uploads ORDER BY uploaded_at DESC"
    if limit:
        c.execute(query + " LIMIT ?", (int(limit),))
    else:
        c.execute(query)
    rows = c.fetchall()
    conn.close()
    return rows


def get_record(id_):
    """Fetch a single record by ID."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (id_,))
    record = c.fetchone()
    conn.close()
    return record


def update_record(id_, ground_truth=None, notes=None):
    """Update record’s ground truth and/or notes fields."""
    if not (ground_truth or notes):
        return
    updates, params = [], []
    if ground_truth is not None:
        updates.append("ground_truth = ?")
        params.append(ground_truth)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)
    conn = get_db_conn()
    c = conn.cursor()
    c.execute(f"UPDATE uploads SET {', '.join(updates)} WHERE id = ?", params + [id_])
    conn.commit()
    conn.close()


def delete_record(id_):
    """Delete a record by ID."""
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM uploads WHERE id = ?", (id_,))
    conn.commit()
    conn.close()


def export_csv_bytes():
    """Export uploads table as CSV in-memory (for download)."""
    import io, csv
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads ORDER BY uploaded_at DESC")
    rows = c.fetchall()
    headers = [d[0] for d in c.description]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows([tuple(r) for r in rows])
    conn.close()
    return output.getvalue().encode("utf-8")


def add_feedback(user_id, rating, comment=None):
    """Insert feedback into feedback table."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (user_id, rating, comment, created_at)
        VALUES (?, ?, ?, ?)
    """, (user_id, rating, comment or "", now))
    conn.commit()
    conn.close()

def ensure_uploads_userid_column():
    """Ensure 'user_id' column exists in uploads table (safe schema migration)."""
    try:
        conn = get_db_conn()
        c = conn.cursor()
        c.execute("PRAGMA table_info(uploads)")
        cols = [r[1] for r in c.fetchall()]
        if "user_id" not in cols:
            c.execute("ALTER TABLE uploads ADD COLUMN user_id INTEGER")
            conn.commit()
            print("Added 'user_id' column to uploads table.")
    except Exception as e:
        print(f"Schema check failed: {e}")
    finally:
        conn.close()


# After app.config and imports
init_db()
ensure_uploads_userid_column()



# -----------------------
# Config
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (225, 225)
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join(BASE_DIR, "models", "mold_model_final.keras")
MODEL = None

# -----------------------
# Load model
# -----------------------
try:
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH, compile=False)
        print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"[WARNING] Model not found at {MODEL_PATH}. Using fallback predictions.")
        MODEL = None
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    MODEL = None

# -----------------------
# Utilities
# -----------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def predict_image_fullpath(path):
    try:
        img = Image.open(path).convert("RGB")  
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  

        if MODEL is not None:
            pred = MODEL.predict(arr, verbose=0)
            pred = np.ravel(pred)
            raw = float(pred[0]) if len(pred) == 1 else float(pred[1])
            label = "mold" if raw >= 0.5 else "clean"
            confidence = float(raw if raw >= 0.5 else 1.0 - raw)
            print(f"[PREDICT] {os.path.basename(path)} → {label} ({confidence:.2f})")
            return label, confidence

        # fallback
        score = (1.0 - arr.mean()) * 0.6 + arr.var() * 0.4
        label = "mold" if score > 0.45 else "clean"
        confidence = float(np.clip(score, 0.0, 1.0))
        return label, confidence

    except Exception as e:
        print(f"[ERROR] Prediction failed for {path}: {e}")
        return "clean", 0.0   

# -----------------------
# Init DB & Model
# -----------------------
import sqlite3
from datetime import datetime, timezone
import os

DB_PATH = "database.db"

def get_db_conn():
    """Return a SQLite connection with dict-style row access and timeout."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize main database tables safely."""
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                failed_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                shap_path TEXT,
                uploaded_at TEXT NOT NULL,
                user_id INTEGER,
                ground_truth TEXT,
                notes TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                rating INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def ensure_uploads_userid_column():
    """Ensure 'user_id' column exists in uploads table (safe schema migration)."""
    try:
        with get_db_conn() as conn:
            c = conn.cursor()
            c.execute("PRAGMA table_info(uploads)")
            cols = [r[1] for r in c.fetchall()]
            if "user_id" not in cols:
                c.execute("ALTER TABLE uploads ADD COLUMN user_id INTEGER")
                conn.commit()
                print("[INFO] Added 'user_id' column to uploads")
    except Exception as e:
        print(f"[ERROR] Schema check failed: {e}")


def init_contacts_table():
    """Ensure contacts table exists."""
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT NOT NULL,
                submitted_at TEXT NOT NULL
            )
        """)
        conn.commit()


# -----------------------
# Call these at app startup
# -----------------------
init_db()
ensure_uploads_userid_column()
init_contacts_table()


# -----------------------
# Contact Form Helpers
# -----------------------
def save_contact(name, email, message):
    """Store contact form submissions in the database."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO contacts (name, email, message, submitted_at) VALUES (?, ?, ?, ?)",
            (name, email, message, now)
        )
        conn.commit()
    print(f"[INFO] Contact saved: {name} <{email}>")


def send_contact_email(name, email, message):
    """Send contact email to admin via SMTP (safe fallback if not configured)."""
    import smtplib
    from email.message import EmailMessage

    ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@moldkit.com")
    SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
    SMTP_USER = os.environ.get("SMTP_USER")
    SMTP_PASS = os.environ.get("SMTP_PASS")

    if not (SMTP_USER and SMTP_PASS):
        print("[WARNING] SMTP credentials not configured. Email not sent.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = "New Contact Form Submission"
        msg["From"] = SMTP_USER
        msg["To"] = ADMIN_EMAIL
        msg.set_content(f"""
Name: {name}
Email: {email}
Message:
{message}
""")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print("[INFO] Contact email sent successfully.")

    except Exception as e:
        print(f"[ERROR] Failed to send contact email: {e}")


# -----------------------
# Forms
# -----------------------
class AdminLoginForm(FlaskForm):
    """Simple admin login form with password validation."""
    password = PasswordField("Password", validators=[DataRequired()])


# -----------------------
# Decorators
# -----------------------
def login_required(f):
    """Ensure user is logged in before accessing protected routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("user_id"):
            return f(*args, **kwargs)
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))
    return wrapper


def admin_required(f):
    """Restrict access to admin-only routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") == "admin":
            return f(*args, **kwargs)
        flash("Admin access only.", "warning")
        return redirect(url_for("admin_login"))
    return wrapper


# -----------------------
# Metrics
# -----------------------
def compute_metrics():
    """
    Compute system-wide stats from prediction records.
    Returns a dictionary containing total uploads, counts, and accuracy if verified.
    """
    rows = list_records()
    total = len(rows)
    mold_count = sum(1 for r in rows if r["prediction"] == "mold")
    clean_count = total - mold_count

    verified = sum(
        1 for r in rows if r["ground_truth"] not in (None, "", "NULL")
    )
    correct = sum(
        1 for r in rows
        if r["ground_truth"] == r["prediction"]
        and r["ground_truth"] not in (None, "", "NULL")
    )

    accuracy = round((correct / verified * 100), 2) if verified else None

    return {
        "total": total,
        "mold": mold_count,
        "clean": clean_count,
        "verified": verified,
        "accuracy": accuracy,
    }


# -----------------------
# Routes - User
# -----------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main landing page for users to upload an image for mold prediction.
    Shows upload form, recent uploads, and key metrics.
    """
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not allowed_file(file.filename):
            flash("Invalid or missing file.", "danger")
            return redirect(url_for("index"))

        # Generate timestamped safe filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        filename = secure_filename(f"{timestamp}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        user_id = session.get("user_id")

        # Run prediction and save record
        label, confidence = predict_image_fullpath(save_path)
        add_record(filename, label, confidence, user_id=user_id)
        log_action("prediction", extra=f"filename={filename}, label={label}, conf={confidence}")

        return redirect(url_for("result_page", filename=filename))

    # GET request → show homepage
    recent_uploads = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "uploaded": r["uploaded_at"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
        }
        for r in list_records(limit=8)
    ]
    metrics = compute_metrics()

    return render_template("index.html", recent=recent_uploads, metrics=metrics)

# -----------------------
# Routes - Predict & Result
# -----------------------
def save_and_predict(file):
    import os
    from werkzeug.utils import secure_filename

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)  # Save file first
        label, confidence = predict_image_fullpath(upload_path)  # Then predict
        return filename, label, confidence
    else:
        raise ValueError("Invalid file uploaded.")
@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "No file uploaded"}), 400

            filename, label, confidence = save_and_predict(file)
            return jsonify({
                "filename": filename,
                "prediction": label,
                "confidence": round(confidence, 2)
            })

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return jsonify({"error": "Prediction failed"}), 500

    # GET request: render the HTML page
    metrics = compute_metrics()
    recent_uploads = [
        {
            "id": r["id"],
            "filename": r["filename"],
            "uploaded": r["uploaded_at"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
        }
        for r in list_records(limit=8)
    ]
    return render_template("predict.html", recent=recent_uploads, metrics=metrics)

@app.route("/result")
@login_required
def result_page():  
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to view your results.", "warning")
        return redirect(url_for("login"))

    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, prediction AS label, confidence, uploaded_at AS upload_date
        FROM uploads
        WHERE user_id = ?
        ORDER BY uploaded_at DESC
        LIMIT 1
    """, (user_id,))
    latest = cursor.fetchone()
    conn.close()

    if not latest:
        flash("No results found yet. Upload an image first.", "info")
        return redirect(url_for("predict_page"))

    latest = dict(latest)

    # Environmental result
    env_result = {
        "weather": "Sunny, 25°C, 60% humidity",
        "prediction": latest['label']
    }

    return render_template(
        "result.html", 
        filename=latest['filename'],
        label=latest['label'],
        confidence=latest['confidence'],
        upload_date=latest['upload_date'],
        result=env_result,
        user=session.get("email")
    )


# -----------------------
# Serve uploaded files
# -----------------------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """
    Serve uploaded files from the UPLOAD_FOLDER.
    Use <path:filename> to allow subfolders if any.
    """
    # Ensure the folder exists
    if not os.path.exists(app.config.get("UPLOAD_FOLDER", "")):
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

# -----------------------
# Feedback Route
# -----------------------
@app.route("/feedback/<filename>", methods=["POST"])
@login_required
def feedback(filename):
    user_id = session.get("user_id")
    if not user_id:
        flash("You must be logged in to submit feedback", "warning")
        return redirect(url_for("login"))

    fb = request.form.get("feedback")
    comment = request.form.get("comment", "")

    # Save feedback in DB
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (user_id, filename, feedback, comment, submitted_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (user_id, filename, fb, comment))
    conn.commit()
    conn.close()

    flash("Thank you for your feedback!", "success")
    return redirect(url_for("result"))

# -----------------------
# User Registration
# -----------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        role = request.form.get("role", "user")  # default to 'user'

        if get_user_by_email(email):
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)
        create_user(email, password_hash, role=role)

        # Auto-login after registration
        user = get_user_by_email(email)
        session["user_id"] = user["id"]
        session["role"] = user["role"]
        session["email"] = user["email"]
        flash(f"Account created. Welcome, {email}!", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


# -----------------------
# User Login
# -----------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = get_user_by_email(email)
        if not user:
            flash("Email not found.", "danger")
            return redirect(url_for("login"))

        # Check if account is locked
        locked_until = user["locked_until"]  # <-- fixed here
        if locked_until:
            from dateutil import parser
            from datetime import datetime, timezone
            locked_time = parser.isoparse(locked_until)
            if datetime.now(timezone.utc) < locked_time:
                flash(f"Account locked until {locked_time.strftime('%H:%M:%S UTC')}", "danger")
                return redirect(url_for("login"))

        # Verify password
        if check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["role"] = user["role"]  # <-- fixed here
            session["email"] = user["email"]
            update_user_login_success(user["id"])
            flash(f"Welcome back, {user['email']}!", "success")
            return redirect(url_for("index"))  # Redirect to homepage
        else:
            increment_failed_attempts(user["id"])
            flash("Incorrect password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

# -----------------------
# Admin Login
# -----------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = get_user_by_email(email)
        if not user:
            flash("Admin email not found.", "danger")
            return redirect(url_for("admin_login"))

        if user["role"] != "admin":
            flash("You are not authorized as admin.", "danger")
            return redirect(url_for("admin_login"))

        if check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["role"] = "admin"
            session["email"] = user["email"]
            update_user_login_success(user["id"])
            flash(f"Welcome Admin, {user['email']}!", "success")
            return redirect(url_for("index"))  # redirect to main page
        else:
            increment_failed_attempts(user["id"])
            flash("Incorrect password.", "danger")
            return redirect(url_for("admin_login"))

    return render_template("admin_login.html")


# -----------------------
# Logout
# -----------------------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# -----------------------
# Profile & Account Management
# -----------------------
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session.get("user_id")
    user = get_user_by_id(user_id) # type: ignore
    if not user:
        flash("User not found", "danger")
        return redirect(url_for("logout"))

    if request.method == "POST":
        new_email = request.form.get("email", "").strip().lower()
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")

        if new_email:
            try:
                set_user_email(user_id, new_email) # type: ignore
                session["email"] = new_email
                flash("Email updated", "success")
            except sqlite3.IntegrityError:
                flash("Email already taken", "danger")

        if current_password and new_password:
            if user.get("password_hash") and check_password_hash(user["password_hash"], current_password):
                set_user_password(user_id, new_password) # type: ignore
                flash("Password updated", "success")
            else:
                flash("Current password incorrect", "danger")

        return redirect(url_for("profile"))

    # Get user's uploads
    uploads = list_records()
    user_uploads = [r for r in uploads if r.get("user_id") == user_id]

    # Pass user as a dict to template
    user_info = {
        "email": user.get("email"),
        "role": user.get("role"),
        "last_login": user.get("last_login")
    }

    return render_template("profile_tabs.html", user=user_info, uploads=user_uploads)


@app.route("/profile/delete_account", methods=["POST"])
@login_required
def delete_account():
    user_id = session.get("user_id")
    conn = get_db_conn()
    cursor = conn.cursor()
    try:
        # Delete user
        cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
        # Optionally nullify uploads' user_id
        cursor.execute("UPDATE uploads SET user_id=NULL WHERE user_id=?", (user_id,))
        conn.commit()
    finally:
        conn.close()

    session.clear()
    flash("Account deleted", "info")
    return redirect(url_for("index"))

# -----------------------
# Admin Dashboard & Management
# -----------------------
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    rows = list_records(limit=200)
    metrics = compute_metrics()
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT 20")
    feedback_rows = c.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", rows=rows, metrics=metrics, feedback=feedback_rows)

@app.route("/admin/users", methods=["GET", "POST"])
@admin_required
def admin_users():
    conn = get_db_conn()
    c = conn.cursor()
    if request.method == "POST":
        action = request.form.get("action")
        target_id = request.form.get("user_id")
        if action == "make_admin":
            c.execute("UPDATE users SET role='admin' WHERE id=?", (target_id,))
        elif action == "remove_admin":
            c.execute("UPDATE users SET role='user' WHERE id=?", (target_id,))
        elif action == "update_email":
            new_email = request.form.get("email", "").strip().lower()
            try:
                c.execute("UPDATE users SET email=? WHERE id=?", (new_email, target_id))
            except sqlite3.IntegrityError:
                flash("Email already taken", "danger")
        conn.commit()
    c.execute("SELECT id,email,role,last_login FROM users ORDER BY id DESC")
    users = c.fetchall()
    conn.close()
    return render_template("admin_users.html", users=users)

@app.route("/admin/delete/<int:record_id>", methods=["POST"])
@admin_required
def admin_delete(record_id):
    conn = get_db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM uploads WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    flash("Record deleted", "info")
    log_action("delete_upload", user_id=session.get("user_id"), extra=f"record_id={record_id}") # type: ignore
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/download_csv")
@admin_required
def admin_download_csv():
    data = export_csv_bytes()
    return send_file(BytesIO(data), as_attachment=True, download_name="uploads.csv", mimetype="text/csv")

# -----------------------
# Public Static Pages
# -----------------------
@app.route("/about")
def about():
    return render_template("about.html", title="About Us")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        # save to database or send email
        flash("Message sent successfully!", "success")
        return redirect(url_for("contact"))

    return render_template("contact.html", title="Contact Us")


# -----------------------
# Password Reset (Stub)
# -----------------------
@app.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = get_user_by_email(email)
        if not user:
            flash("If that email exists, a reset link was sent (stub).", "info")
            return redirect(url_for("login"))
        token = secrets.token_urlsafe(32)
        flash("Password reset flow triggered (stub).", "info")
        return redirect(url_for("login"))
    return render_template("forgot.html")

@app.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    if request.method == "POST":
        new_pw = request.form.get("new_password", "")
        confirm = request.form.get("confirm", "")
        if not new_pw or new_pw != confirm:
            flash("Passwords missing or do not match.", "warning")
            return redirect(url_for("reset_password", token=token))
        flash("Password reset (stub). Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("reset.html", token=token)


@app.route("/profile_settings", methods=["GET", "POST"])
@login_required
def profile_settings():
    """View and update profile settings."""
    user_id = session.get("user_id")

    if not user_id:
        flash("Please log in to manage your profile.", "warning")
        return redirect(url_for("login"))

    try:
        conn = get_db_conn()
        cursor = conn.cursor()

        # Handle update
        if request.method == "POST":
            new_email = request.form.get("email")
            new_password = request.form.get("password")

            if new_password:
                cursor.execute(
                    "UPDATE users SET email=?, password_hash=? WHERE id=?",
                    (new_email, generate_password_hash(new_password), user_id)
                )
            else:
                cursor.execute(
                    "UPDATE users SET email=? WHERE id=?",
                    (new_email, user_id)
                )

            conn.commit()
            flash("Profile updated successfully!", "success")
            session["email"] = new_email
            conn.close()
            return redirect(url_for("profile_settings"))

        # Display profile info
        cursor.execute("SELECT email FROM users WHERE id=?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            flash("User not found.", "danger")
            return redirect(url_for("index"))

        return render_template("profile_settings.html", user=user)

    except Exception as e:
        print(f"Profile settings error: {e}")
        flash("An error occurred while loading profile settings.", "danger")
        return redirect(url_for("index"))

from datetime import datetime

@app.route("/privacy")
def privacy():
    """
    Render the Privacy Policy page with a dynamic last-updated date.
    """
    last_updated = datetime.now().strftime("%B %d, %Y")  
    return render_template("privacy.html", last_updated=last_updated)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    # --- Validate input ---
    if "image" not in request.files:
        flash("No image uploaded", "danger")
        return redirect(url_for("predict_page"))

    file = request.files["image"]
    location = request.form.get("location", "").strip()
    ventilation = request.form.get("ventilation", "").lower()
    structure = request.form.get("structure", "").lower()
    leak = request.form.get("leak", "").lower()
    health = request.form.get("health", "").lower()

    # --- Save & Predict ---
    filename = save_and_predict(file)  # Existing function
    label, confidence = get_prediction(filename)

    # --- Fetch real-time climate data ---
    weather_data = "Unavailable"
    humidity = temperature = 0
    try:
        if location:
            res = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": location, "appid": OPENWEATHER_API_KEY, "units": "metric"}, # type: ignore
                timeout=5
            )
            if res.ok:
                data = res.json()
                temperature = data["main"].get("temp", 0)
                humidity = data["main"].get("humidity", 0)
                weather_data = f"{temperature}°C, {humidity}% humidity"
            else:
                weather_data = "Location not found"
        else:
            weather_data = "No location provided"
    except Exception as e:
        weather_data = f"Error fetching weather data: {e}"

    # --- Environmental Risk Scoring ---
    risk_score = 0
    ventilation_risk = {"poor": 2, "moderate": 1, "good": 0}
    risk_score += ventilation_risk.get(ventilation, 0)
    if leak == "yes":
        risk_score += 2
    if humidity > 70:
        risk_score += 2
    elif humidity > 50:
        risk_score += 1
    if health == "yes":
        risk_score += 1

    # --- Combine ML + Environmental Assessment ---
    if label == "clean" and risk_score >= 4:
        final_status = "Mold likely to appear soon (poor conditions)"
    elif label == "mold" and risk_score <= 2:
        final_status = "Mold detected, environment aggravating"
    elif label == "clean":
        final_status = "Clean and low risk"
    else:
        final_status = "Mold present, take immediate action"

    # --- Save results in DB ---
    save_prediction(filename, label, confidence, location, weather_data, final_status) # type: ignore

    # --- Render results page ---
    env_result = {
        "weather": weather_data,
        "temperature": temperature,
        "humidity": humidity,
        "prediction": final_status
    }

    return render_template(
        "result.html",
        filename=filename,
        label=label,
        confidence=confidence,
        upload_date=None,  # Optional: could store actual timestamp
        result=env_result,
        user=session.get("email")
    )


# ----------------------
# Printable Report Route
# ----------------------
@app.route("/print_report/<filename>")
def print_report(filename):
    label = request.args.get("label", "unknown")
    confidence = float(request.args.get("confidence", 0))
    weather = request.args.get("weather", "Unavailable")
    prediction = request.args.get("prediction", "Unavailable")

    return render_template(
        "print_report.html",
        filename=filename,
        label=label,
        confidence=confidence,
        weather=weather,
        prediction=prediction
    )


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
