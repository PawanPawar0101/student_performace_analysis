import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def generate_synthetic_data(n_samples=3000):
    """
    Generate realistic synthetic student performance data.
    Explicitly constructs each grade class to ensure all 5 are represented.
    """
    np.random.seed(42)
    data = []

    # Define grade bands: 0=A(90-100), 1=B(75-89), 2=C(60-74), 3=D(45-59), 4=F(<45)
    # Distribute samples across classes so every class is well-represented
    class_configs = [
        # (grade_cat, target_score_mean, target_score_std, n)
        (0, 93, 4,  int(n_samples * 0.15)),   # A — 15%
        (1, 81, 5,  int(n_samples * 0.30)),   # B — 30%
        (2, 67, 5,  int(n_samples * 0.30)),   # C — 30%
        (3, 52, 4,  int(n_samples * 0.15)),   # D — 15%
        (4, 35, 6,  int(n_samples * 0.10)),   # F — 10%
    ]

    for grade_cat, score_mean, score_std, n in class_configs:
        for _ in range(n):
            # Back-calculate plausible features from the target score band
            # Higher score students tend to study more, attend more, etc.
            score_factor = (score_mean - 35) / 60.0  # 0.0 → 1.0

            study_hours  = np.clip(np.random.normal(score_factor * 10 + 1, 1.5), 0, 12)
            attendance   = np.clip(np.random.normal(score_factor * 50 + 45, 8), 30, 100)
            prev_grades  = np.clip(np.random.normal(score_mean - 3, 8), 10, 100)
            sleep_hours  = np.clip(np.random.normal(7 if score_factor > 0.4 else 5.5, 1.2), 3, 12)
            absences     = int(np.clip(np.random.exponential((1 - score_factor) * 10 + 1), 0, 30))
            motivation   = np.random.choice([1,2,3,4,5], p=_motivation_dist(score_factor))
            health       = np.random.choice([1,2,3,4,5], p=_health_dist(score_factor))
            parental_edu = np.random.choice([0,1,2,3,4], p=_edu_dist(score_factor))
            parental_sup = np.random.choice([0,1,2,3,4], p=_support_dist(score_factor))
            internet     = 1 if score_factor > 0.3 else np.random.choice([0,1])
            extracurr    = 1 if score_factor > 0.5 else np.random.choice([0,1])
            tutoring     = 1 if score_factor < 0.4 else np.random.choice([0,1])
            age          = np.random.randint(16, 25)
            gender       = np.random.choice([0, 1])

            data.append([
                age, gender, study_hours, attendance, prev_grades,
                parental_edu, internet, extracurr,
                sleep_hours, health, absences, tutoring,
                parental_sup, motivation, grade_cat
            ])

    cols = [
        'age', 'gender', 'study_hours', 'attendance', 'prev_grades',
        'parental_education', 'internet_access', 'extracurricular',
        'sleep_hours', 'health', 'absences', 'tutoring',
        'parental_support', 'motivation', 'grade_cat'
    ]
    df = pd.DataFrame(data, columns=cols)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def _motivation_dist(f):
    # f close to 1 → skewed toward 4/5; close to 0 → skewed toward 1/2
    if f > 0.7:   return [0.02, 0.05, 0.15, 0.40, 0.38]
    elif f > 0.4: return [0.05, 0.10, 0.35, 0.35, 0.15]
    elif f > 0.2: return [0.10, 0.30, 0.35, 0.20, 0.05]
    else:         return [0.35, 0.35, 0.20, 0.07, 0.03]

def _health_dist(f):
    if f > 0.7:   return [0.02, 0.05, 0.18, 0.40, 0.35]
    elif f > 0.4: return [0.05, 0.12, 0.33, 0.35, 0.15]
    elif f > 0.2: return [0.10, 0.25, 0.38, 0.20, 0.07]
    else:         return [0.30, 0.35, 0.25, 0.07, 0.03]

def _edu_dist(f):
    if f > 0.7:   return [0.02, 0.10, 0.28, 0.35, 0.25]
    elif f > 0.4: return [0.05, 0.18, 0.38, 0.28, 0.11]
    elif f > 0.2: return [0.10, 0.28, 0.38, 0.18, 0.06]
    else:         return [0.20, 0.35, 0.30, 0.12, 0.03]

def _support_dist(f):
    if f > 0.7:   return [0.02, 0.08, 0.20, 0.38, 0.32]
    elif f > 0.4: return [0.05, 0.15, 0.35, 0.30, 0.15]
    elif f > 0.2: return [0.12, 0.28, 0.35, 0.18, 0.07]
    else:         return [0.28, 0.35, 0.25, 0.09, 0.03]


def train_model(model_path):
    """Train Random Forest model and save it."""
    print("Generating training data...")
    df = generate_synthetic_data(3000)

    feature_cols = [
        'age', 'gender', 'study_hours', 'attendance', 'prev_grades',
        'parental_education', 'internet_access', 'extracurricular',
        'sleep_hours', 'health', 'absences', 'tutoring',
        'parental_support', 'motivation'
    ]

    X = df[feature_cols].values
    y = df['grade_cat'].values

    print(f"Class distribution: { {int(k): int(v) for k,v in zip(*np.unique(y, return_counts=True))} }")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,          # let trees grow fully
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced', # handles class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Safe classification report — only report classes present in test set
    present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    grade_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    target_names = [grade_names[c] for c in present_classes]

    print(classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=target_names,
        zero_division=0
    ))

    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'accuracy': accuracy
    }, model_path)
    print(f"Model saved to {model_path}")
    return model


def predict_performance(features_dict, model_path):
    """Predict student performance given a feature dict."""
    model_data = joblib.load(model_path)
    model      = model_data['model']
    feature_cols = model_data['feature_cols']

    X = np.array([[features_dict[col] for col in feature_cols]])

    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    # Map class index → probability (model may not have trained on all 5)
    classes = list(model.classes_)
    conf = float(pred_proba[classes.index(pred_class)]) * 100 if pred_class in classes else 0.0

    # Weighted score for the progress bar (purely for display)
    f = features_dict
    raw_score = (
        0.25 * (f['study_hours']  / 12   * 100) +
        0.20 * f['attendance']                   +
        0.20 * f['prev_grades']                  +
        0.08 * (f['parental_education'] / 4 * 100) +
        0.05 * (f['internet_access']    * 100)  +
        0.04 * (f['extracurricular']    * 100)  +
        0.05 * (f['sleep_hours']  / 12  * 100)  +
        0.04 * (f['health']       / 5   * 100)  +
        0.04 * (f['tutoring']     * 100)         +
        0.03 * (f['parental_support'] / 4 * 100)+
        0.06 * (f['motivation']   / 5   * 100)  +
       -0.06 *  f['absences']
    )
    raw_score = max(0, min(100, raw_score))

    grade_map    = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    category_map = {0: 'Outstanding', 1: 'Good', 2: 'Average', 3: 'Below Average', 4: 'Failing'}

    return {
        "grade":            grade_map.get(pred_class, '?'),
        "category":         category_map.get(pred_class, 'Unknown'),
        "confidence":       round(conf, 2),
        "score":            round(raw_score, 2),
        "recommendations":  _recommendations(features_dict, pred_class),
        "feature_importance": _feature_importance(model, feature_cols)
    }


def _recommendations(f, pred_class):
    recs = []
    if f['study_hours'] < 4:
        recs.append("📚 Increase daily study hours to at least 4–6 hours for better retention.")
    if f['attendance'] < 75:
        recs.append("🏫 Improve class attendance above 75%. It is the strongest predictor of success.")
    if f['sleep_hours'] < 6 or f['sleep_hours'] > 9:
        recs.append("😴 Aim for 7–8 hours of sleep. Good sleep consolidates memory.")
    if f['absences'] > 10:
        recs.append("📅 Reduce absences — you've missed too many classes this semester.")
    if f['extracurricular'] == 0:
        recs.append("🏃 Consider joining at least one extracurricular activity for holistic growth.")
    if f['tutoring'] == 0 and pred_class >= 2:
        recs.append("👨‍🏫 Consider tutoring sessions to strengthen weak subjects.")
    if f['motivation'] < 3:
        recs.append("🎯 Set short-term goals and reward yourself — motivation is learnable.")
    if f['health'] < 3:
        recs.append("🏥 Prioritise your health — physical wellbeing directly impacts academic performance.")
    if not recs:
        recs.append("🌟 Excellent habits! Keep up the great work.")
        recs.append("🚀 Challenge yourself with advanced topics and research opportunities.")
    return recs


def _feature_importance(model, feature_cols):
    importance = model.feature_importances_
    items = [
        {"feature": col.replace("_", " ").title(), "importance": round(float(imp) * 100, 2)}
        for col, imp in zip(feature_cols, importance)
    ]
    return sorted(items, key=lambda x: x["importance"], reverse=True)[:6]