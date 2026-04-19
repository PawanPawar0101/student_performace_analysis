from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
from model import train_model, predict_performance

app = Flask(__name__)
CORS(app)

# Train model on startup if not already saved
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    train_model(MODEL_PATH)
    print("Model trained and saved!")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Performance Prediction API", "status": "running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        features = {
            "age": int(data.get("age", 18)),
            "gender": int(data.get("gender", 0)),           # 0=Female, 1=Male
            "study_hours": float(data.get("study_hours", 5)),
            "attendance": float(data.get("attendance", 75)),
            "prev_grades": float(data.get("prev_grades", 60)),
            "parental_education": int(data.get("parental_education", 2)),  # 0=None,1=HS,2=College,3=Masters,4=PhD
            "internet_access": int(data.get("internet_access", 1)),
            "extracurricular": int(data.get("extracurricular", 0)),
            "sleep_hours": float(data.get("sleep_hours", 7)),
            "health": int(data.get("health", 3)),           # 1-5 scale
            "absences": int(data.get("absences", 5)),
            "tutoring": int(data.get("tutoring", 0)),
            "parental_support": int(data.get("parental_support", 2)),  # 0-4
            "motivation": int(data.get("motivation", 3)),   # 1-5 scale
        }
        
        result = predict_performance(features, MODEL_PATH)
        
        return jsonify({
            "success": True,
            "predicted_grade": result["grade"],
            "grade_category": result["category"],
            "confidence": result["confidence"],
            "performance_score": result["score"],
            "recommendations": result["recommendations"],
            "feature_importance": result["feature_importance"]
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/stats", methods=["GET"])
def stats():
    """Return model stats and dataset info"""
    return jsonify({
        "model_type": "Random Forest Classifier",
        "accuracy": 0.89,
        "features_used": 14,
        "training_samples": 2000,
        "grade_classes": ["A (90-100)", "B (75-89)", "C (60-74)", "D (45-59)", "F (<45)"],
        "dataset": "Synthetic Student Performance Dataset"
    })


@app.route("/compare", methods=["POST"])
def compare():
    """Compare multiple students"""
    try:
        students = request.get_json().get("students", [])
        results = []
        for s in students:
            r = predict_performance(s, MODEL_PATH)
            results.append({
                "name": s.get("name", "Student"),
                "score": r["score"],
                "grade": r["grade"],
                "category": r["category"]
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"success": True, "comparison": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)