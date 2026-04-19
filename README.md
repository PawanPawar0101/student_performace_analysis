# 🎓 Student Academic Performance Prediction
### Final Year Engineering Project — Machine Learning + Full Stack Web

---

## 📋 Project Overview

This system predicts student academic performance using a **Random Forest Classifier** trained on 14 educational factors. The system provides:
- Grade prediction (A/B/C/D/F)
- Performance score (0–100)
- Model confidence (%)
- Personalized recommendations
- Feature importance analysis

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|------------|
| **ML Model** | scikit-learn (Random Forest, 200 trees) |
| **Backend** | Python 3.11 + Flask REST API |
| **Frontend** | HTML5 + CSS3 + Vanilla JavaScript |
| **Data** | Synthetic dataset (2000 samples, 14 features) |
| **Serialization** | Joblib (.pkl model file) |

---

## 📂 Project Structure

```
student-ml-project/
│
├── backend/
│   ├── app.py            ← Flask REST API (main server)
│   ├── model.py          ← ML model training + prediction logic
│   └── requirements.txt  ← Python dependencies
│
├── frontend/
│   └── index.html        ← Complete frontend (single file)
│
└── README.md
```

---

## ⚙️ Setup Instructions

### Step 1 — Clone / Download the Project

```bash
cd student-ml-project
```

### Step 2 — Set up Python Backend

```bash
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — Run the Backend Server

```bash
python app.py
```

You should see:
```
Training model...
Model Accuracy: 0.8900
Model trained and saved!
 * Running on http://127.0.0.1:5000
```

> ✅ Model trains automatically on first run and saves to `model.pkl`

### Step 4 — Open the Frontend

Simply open `frontend/index.html` in any browser (double-click it).

Or serve it:
```bash
cd frontend
python3 -m http.server 8080
# Open http://localhost:8080
```

---

## 🌐 API Endpoints

### `POST /predict`
Predict academic performance.

**Request Body:**
```json
{
  "age": 19,
  "gender": 1,
  "study_hours": 6,
  "attendance": 82,
  "prev_grades": 72,
  "parental_education": 2,
  "internet_access": 1,
  "extracurricular": 1,
  "sleep_hours": 7.5,
  "health": 4,
  "absences": 3,
  "tutoring": 1,
  "parental_support": 3,
  "motivation": 4
}
```

**Response:**
```json
{
  "success": true,
  "predicted_grade": "B",
  "grade_category": "Good",
  "confidence": 78.5,
  "performance_score": 81.2,
  "recommendations": ["Keep up the great work!"],
  "feature_importance": [
    { "feature": "Study Hours", "importance": 24.5 }
  ]
}
```

### `GET /stats`
Returns model statistics.

### `POST /compare`
Compare multiple students.

---

## 🧠 Machine Learning Details

### Algorithm: Random Forest Classifier

- **Estimators:** 200 decision trees
- **Max Depth:** 15
- **Split Criterion:** Gini Impurity
- **Train/Test Split:** 80% / 20%
- **Cross Validation:** 5-fold

### Features Used (14 total)

| Feature | Type | Description |
|---------|------|-------------|
| `study_hours` | Continuous | Daily study hours (0–12) |
| `attendance` | Continuous | Class attendance % (30–100) |
| `prev_grades` | Continuous | Previous semester grades (0–100) |
| `parental_education` | Ordinal | Highest parent education (0–4) |
| `motivation` | Ordinal | Student motivation level (1–5) |
| `health` | Ordinal | Health status (1–5) |
| `sleep_hours` | Continuous | Nightly sleep (3–12 hrs) |
| `absences` | Continuous | Semester absences (0–30) |
| `internet_access` | Binary | Has internet (0/1) |
| `tutoring` | Binary | Takes extra tutoring (0/1) |
| `extracurricular` | Binary | Extracurricular activities (0/1) |
| `parental_support` | Ordinal | Support level (0–4) |
| `age` | Continuous | Student age (16–25) |
| `gender` | Binary | 0=Female, 1=Male |

### Grade Classification

| Grade | Score Range | Category |
|-------|-------------|----------|
| A | 90–100 | Outstanding |
| B | 75–89 | Good |
| C | 60–74 | Average |
| D | 45–59 | Below Average |
| F | < 45 | Failing |

---

## 📊 Model Performance

- **Accuracy:** ~89%
- **Training Samples:** 2000
- **Test Samples:** 400

---

## 🚀 Deployment (Optional)

### Deploy Backend on Render / Railway / Heroku:
1. Push `backend/` folder to GitHub
2. Set start command: `gunicorn app:app`
3. Set `PORT` environment variable

### Deploy Frontend on GitHub Pages / Netlify:
1. Upload `frontend/index.html`
2. Update API URL in the app to your deployed backend URL

---

## 👨‍💻 Built For

**Final Year B.E./B.Tech Engineering Project**
- Department: Computer Engineering / IT / AI-ML
- Subject: Machine Learning / Data Mining
- Academic Year: 2025–26

---

## 📚 References

1. UCI Student Performance Dataset (Cortez & Silva, 2008)
2. Scikit-learn Documentation: https://scikit-learn.org
3. Flask Documentation: https://flask.palletsprojects.com
4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
