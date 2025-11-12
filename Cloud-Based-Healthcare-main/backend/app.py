import os
import io
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import cloudpickle
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')  # Prevent GUI-related warnings
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# =========================
# EXTRA VISUALIZATION FUNCTION
# =========================

def generate_signal_chart(disease_type, data=None):
    """Generate ECG-like or glucose marker chart image dynamically."""
    import numpy as np
    plt.figure(figsize=(4, 1.5))

    if "heart" in disease_type.lower():
        # Simulated ECG for heart patients
        t = np.linspace(0, 2*np.pi, 200)
        ecg = np.sin(5*t) * (np.sin(2*t) > 0)
        plt.plot(t, ecg, color="#d60000", linewidth=1.8)
        plt.title("ECG Trend", color="#0056A4", fontsize=9)
        plt.axis('off')

    elif "diabet" in disease_type.lower():
        # Bar chart for diabetes markers
        h = float(data.get("Hemoglobin (g/dL)", 0) or 0)
        p = float(data.get("Urine Protein (mg/dL)", 0) or 0)
        g = float(data.get("Urine Glucose (mg/dL)", 0) or 0)

        labels = ["Hemoglobin", "Urine Protein", "Urine Glucose"]
        values = [h, p, g]
        colors = ["#0078FF", "#00A36C", "#E63946"]

        plt.bar(labels, values, color=colors)
        plt.title("Key Diabetes Marker Levels", color="#0056A4", fontsize=9)
        plt.ylabel("Value")
        plt.grid(axis="y", linestyle="--", alpha=0.4)

    else:
        # Default flat trend
        x = np.arange(10)
        y = np.ones(10) * 5
        plt.plot(x, y, color="#0078FF", linewidth=2)
        plt.title("Stability Trend", color="#0056A4", fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    img_path = os.path.join(BASE_DIR, "signal_chart.png")
    plt.savefig(img_path, transparent=True)
    plt.close()
    return img_path



# =========================
# BASE & STAFFING CSV LOAD
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAFFING_PATH = os.path.join(BASE_DIR, "staffing_simulation_summary.csv")


def load_staffing():
    """
    Load staffing_simulation_summary.csv with columns:
    Date,Beds,Nurses,Doctors,Unit
    """
    try:
        df = pd.read_csv(STAFFING_PATH)
        expected_cols = {"Date", "Beds", "Nurses", "Doctors", "Unit"}
        if not expected_cols.issubset(df.columns):
            print(f"[WARN] Staffing CSV missing expected columns. Found: {df.columns.tolist()}")
            return pd.DataFrame(columns=list(expected_cols))

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        print(f"[INFO] Loaded staffing CSV with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load staffing CSV: {e}")
        return pd.DataFrame(columns=["Date", "Beds", "Nurses", "Doctors", "Unit"])


STAFFING_DF = None

# =========================================================
# FOLLOW-UP DATABASE (CSV) SETUP
# =========================================================
FOLLOWUP_PATH = os.path.join(BASE_DIR, "patient_followups.csv")

# Ensure the file exists
if not os.path.exists(FOLLOWUP_PATH):
    pd.DataFrame(columns=[
        "Patient ID", "Patient Name", "Problem Type",
        "Readmission Probability", "Risk Label",
        "Followup Channel", "Next Visit",
        "Simulation Date", "Hospital Unit",
        "Prediction Date", "Status"
    ]).to_csv(FOLLOWUP_PATH, index=False)


def save_followup_record(record):
    """Append one patient follow-up record to CSV."""
    df = pd.read_csv(FOLLOWUP_PATH)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(FOLLOWUP_PATH, index=False)
    print(f"[INFO] Saved follow-up for {record.get('Patient Name')}")


# =========================
# FLASK APP CONFIG
# =========================

app = Flask(__name__, static_folder="frontend", static_url_path="/")
CORS(app)

# =========================
# FEATURES (DO NOT CHANGE)
# =========================

COMMON_FEATURES = [
    "Age",
    "Sex",
    "Weight",
    "Blood Pressure",
    "Cholesterol",
    "Insulin",
    "Platelets",
    "Diabetics",
    "air_quality_index",
    "social_event_count",
]

DIABETES_FEATURES = [
    "Hemoglobin (g/dL)",
    "WBC Count (10^9/L)",
    "Platelet Count (10^9/L)",
    "Urine Protein (mg/dL)",
    "Urine Glucose (mg/dL)",
]

HEART_FAILURE_FEATURES = [
    "ECG Result",
    "Pulse Rate (bpm)",
]

ordinal_map = {
    "Low": 1,
    "Normal": 2,
    "Moderate": 3,
    "High": 4,
    "low": 1,
    "normal": 2,
    "moderate": 3,
    "high": 4,
    "Borderline": 3,
    "borderline": 3,
    "Abnormal": 4,
    "abnormal": 4,
}

# =========================
# COMPAT FOR OLD PIPELINES
# =========================

class BloodPressureTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


sys.modules["__main__"].BloodPressureTransformer = BloodPressureTransformer

# =========================
# MODEL LOADER
# =========================

def load_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                print(f"[MODEL] Loading via joblib: {p}")
                return joblib.load(p)
            except Exception as e:
                print(f"[WARN] joblib.load failed for {p}: {e}")
                print("[INFO] Trying cloudpickle fallback...")
                with open(p, "rb") as f:
                    return cloudpickle.load(f)
    raise FileNotFoundError(f"No model file found in: {paths}")


diabetes_model = load_first_existing(
    [
        "readmission_diabetes_RandomForest.pkl",
        "readmission_diabetes_advanced.pkl",
        "readmission_diabetes.pkl",
    ]
)

heart_model = load_first_existing(
    [
        "readmission_heart_disease_RandomForest.pkl",
        "readmission_heart_disease_advanced.pkl",
        "readmission_heart_disease.pkl",
    ]
)

# =========================
# HELPER FUNCTIONS
# =========================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        s = str(value).strip()
        if s == "" or s.lower() in ["nan", "none", "null"]:
            return default
        return float(s)
    except Exception:
        return default


def encode_ordinal(value, default=2):
    if value is None:
        return default
    return ordinal_map.get(str(value).strip(), default)


def encode_bp(bp_str):
    try:
        s, d = str(bp_str).split("/")
        return float(s) / 120.0 + float(d) / 80.0
    except Exception:
        return 2.0


def compute_severity_score(payload, disease):
    try:
        age = safe_float(payload.get("Age", 0))
        bp = str(payload.get("Blood Pressure", "120/80"))
        chol = safe_float(payload.get("Cholesterol", 0))
        insulin = str(payload.get("Insulin", "Normal")).lower()
        diab = str(payload.get("Diabetics", "Normal")).lower()
        aqi = safe_float(payload.get("air_quality_index", 50))
        events = safe_float(payload.get("social_event_count", 0))
        hgb = safe_float(payload.get("Hemoglobin (g/dL)", 13.5))
        wbc = safe_float(payload.get("WBC Count (10^9/L)", 7.0))
        uprot = safe_float(payload.get("Urine Protein (mg/dL)", 10))
        uglu = safe_float(payload.get("Urine Glucose (mg/dL)", 5))
        ecg = str(payload.get("ECG Result", "Normal")).lower()
        pulse = safe_float(payload.get("Pulse Rate (bpm)", 72))
    except Exception:
        return 0.0

    score = 0.0

    if age >= 75:
        score += 2
    elif age >= 60:
        score += 1

    try:
        s, d = bp.split("/")
        s = float(s)
        d = float(d)
        if s >= 160 or d >= 100:
            score += 2
        elif s >= 140 or d >= 90:
            score += 1
    except Exception:
        pass

    if chol >= 260:
        score += 2
    elif chol >= 220:
        score += 1

    if "high" in insulin:
        score += 1
    if "high" in diab:
        score += 2
    if uprot >= 30:
        score += 1
    if uglu >= 20:
        score += 1

    if wbc >= 11:
        score += 1
    if hgb < 10:
        score += 1

    if disease == "Heart Disease":
        if "abnormal" in ecg:
            score += 2
        elif "borderline" in ecg:
            score += 1
        if pulse >= 100:
            score += 1

    if aqi >= 120:
        score += 1
    if events >= 3:
        score += 0.5

    return score


def adjusted_risk_score(model_prob, payload, disease):
    sev = compute_severity_score(payload, disease)
    sev_norm = max(0.0, min(sev, 8.0)) / 8.0
    combined = 0.4 * float(model_prob) + 0.6 * sev_norm
    return max(0.0, min(combined, 1.0))


def risk_category(adj_prob):
    if adj_prob < 0.40:
        return "Low"
    elif adj_prob < 0.70:
        return "Medium"
    return "High"


def followup_plan(risk_score, problem_type):
    if risk_score >= 0.7:
        return {
            "risk_band": "High",
            "channel": "Phone + SMS + App",
            "schedule": ["48 hours", "7 days", "14 days"],
            "note": "High risk of readmission. Arrange follow-up within 2 days.",
        }
    elif risk_score >= 0.4:
        return {
            "risk_band": "Medium",
            "channel": "SMS + App",
            "schedule": ["5 days", "14 days"],
            "note": "Moderate risk. Review within 4–5 days.",
        }
    else:
        return {
            "risk_band": "Low",
            "channel": "Portal / Email",
            "schedule": ["14 days"],
            "note": "Low risk. Routine follow-up in 1–2 weeks.",
        }

# =========================
# STAFFING SIMULATOR (CSV)
# =========================

def staffing_simulator(risk_score, sim_date=None, hospital_unit=None):
    if STAFFING_DF is None or STAFFING_DF.empty:
        expected = round(risk_score * 10, 2)
        base = max(1, int(expected))
        return {
            "expected_readmissions": expected,
            "suggested_beds": base,
            "suggested_nurses": max(1, base // 2 + 1),
            "suggested_doctors": max(1, base // 3 + 1),
        }

    df = STAFFING_DF.copy()

    if hospital_unit:
        df = df[df["Unit"].astype(str).str.contains(str(hospital_unit), case=False, na=False)]

    if sim_date:
        try:
            sim_dt = pd.to_datetime(sim_date, errors="coerce")
            if not pd.isna(sim_dt):
                df["date_diff"] = (df["Date"] - sim_dt).abs().dt.days
                df = df.sort_values("date_diff")
                df = df[df["date_diff"] <= 3]
        except Exception as e:
            print(f"[WARN] Bad sim_date: {e}")

    if df.empty:
        df = STAFFING_DF.copy()

    beds_base = max(1, int(round(df["Beds"].mean())))
    nurses_base = max(1, int(round(df["Nurses"].mean())))
    doctors_base = max(1, int(round(df["Doctors"].mean())))

    factor = 0.8 + risk_score * 0.8
    beds = max(1, int(round(beds_base * factor)))
    nurses = max(1, int(round(nurses_base * factor)))
    doctors = max(1, int(round(doctors_base * factor)))

    expected = round(risk_score * 10, 2)

    return {
        "expected_readmissions": expected,
        "suggested_beds": beds,
        "suggested_nurses": nurses,
        "suggested_doctors": doctors,
    }

# =========================
# BUILD FEATURE DATAFRAME
# =========================

def build_feature_df(payload):
    problem_type = (payload.get("Problem Type") or "").strip()
    problem_type_lower = problem_type.lower()

    if "diab" in problem_type_lower:
        features = COMMON_FEATURES + DIABETES_FEATURES
        model = diabetes_model
        disease = "Diabetes"
    else:
        features = COMMON_FEATURES + HEART_FAILURE_FEATURES
        model = heart_model
        disease = "Heart Disease"

    row = {}

    row["Age"] = safe_float(payload.get("Age", 0))
    sex_raw = (payload.get("Sex") or "Male").strip().lower()
    row["Sex"] = 1.0 if sex_raw == "female" else 0.0
    row["Weight"] = safe_float(payload.get("Weight", 0))
    row["Blood Pressure"] = encode_bp(payload.get("Blood Pressure", "120/80"))
    row["Cholesterol"] = safe_float(payload.get("Cholesterol", 0))
    row["Insulin"] = float(encode_ordinal(payload.get("Insulin", "Normal")))
    row["Platelets"] = safe_float(payload.get("Platelets", 0))
    row["Diabetics"] = float(encode_ordinal(payload.get("Diabetics", "Normal")))
    row["air_quality_index"] = safe_float(payload.get("air_quality_index", 50))
    row["social_event_count"] = safe_float(payload.get("social_event_count", 0))

    if disease == "Diabetes":
        row["Hemoglobin (g/dL)"] = safe_float(payload.get("Hemoglobin (g/dL)", 13.5))
        row["WBC Count (10^9/L)"] = safe_float(payload.get("WBC Count (10^9/L)", 7.0))
        row["Platelet Count (10^9/L)"] = safe_float(payload.get("Platelet Count (10^9/L)", 250))
        row["Urine Protein (mg/dL)"] = safe_float(payload.get("Urine Protein (mg/dL)", 10))
        row["Urine Glucose (mg/dL)"] = safe_float(payload.get("Urine Glucose (mg/dL)", 5))
    else:
        row["ECG Result"] = float(encode_ordinal(payload.get("ECG Result", "Normal")))
        row["Pulse Rate (bpm)"] = safe_float(payload.get("Pulse Rate (bpm)", 72))

    full_features = COMMON_FEATURES + (
        DIABETES_FEATURES if disease == "Diabetes" else HEART_FAILURE_FEATURES
    )
    values = [safe_float(row.get(col, 0.0), 0.0) for col in full_features]
    X = pd.DataFrame([values], columns=full_features).astype("float64")

    return X, model, disease

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        X, model, disease = build_feature_df(data)
        model_prob = float(model.predict_proba(X)[0, 1])
        adj_prob = adjusted_risk_score(model_prob, data, disease)
        risk = risk_category(adj_prob)
        followup = followup_plan(adj_prob, disease)

        sim_date = data.get("Simulation Date")
        hospital_unit = data.get("Hospital Unit")
        staffing = staffing_simulator(adj_prob, sim_date=sim_date, hospital_unit=hospital_unit)

        final_pred = "Yes" if adj_prob >= 0.5 else "No"

        from datetime import datetime

        record = {
            "Patient ID": data.get("Patient ID", "N/A"),
            "Patient Name": data.get("Patient Name", "N/A"),
            "Problem Type": disease,
            "Readmission Probability": round(adj_prob, 4),
            "Risk Label": risk,
            "Followup Channel": followup["channel"],
            "Next Visit": followup["schedule"][0] if isinstance(followup.get("schedule"), list) else "N/A",
            "Simulation Date": sim_date or "N/A",
            "Hospital Unit": hospital_unit or "N/A",
            "Prediction Date": datetime.now().strftime("%Y-%m-%d"),
            "Status": "Pending"
        }
        save_followup_record(record)

        return jsonify(
            {
                "disease_type": disease,
                "readmission_probability": round(adj_prob, 4),
                "prediction": final_pred,
                "risk_label": risk,
                "followup_plan": followup,
                "staffing": staffing,
            }
        )
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate_staffing", methods=["POST"])
def api_simulate_staffing():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        X, model, disease = build_feature_df(data)
        model_prob = float(model.predict_proba(X)[0, 1])
        adj_prob = adjusted_risk_score(model_prob, data, disease)

        sim_date = data.get("Simulation Date")
        hospital_unit = data.get("Hospital Unit")
        staffing = staffing_simulator(adj_prob, sim_date=sim_date, hospital_unit=hospital_unit)

        return jsonify(
            {
                "simulation_date": sim_date or "N/A",
                "hospital_unit": hospital_unit or "N/A",
                "risk_score": round(adj_prob, 4),
                "staffing": staffing,
            }
        )
    except Exception as e:
        print(f"[ERROR] Staffing simulation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def api_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data"}), 400

        X, model, disease = build_feature_df(data)
        model_prob = float(model.predict_proba(X)[0, 1])
        adj_prob = adjusted_risk_score(model_prob, data, disease)
        risk = risk_category(adj_prob)
        followup = followup_plan(adj_prob, disease)

        sim_date = data.get("Simulation Date")
        hospital_unit = data.get("Hospital Unit")
        staffing = staffing_simulator(adj_prob, sim_date=sim_date, hospital_unit=hospital_unit)

                # ================== PDF DESIGN START ==================
                # ================== PDF DESIGN START ==================
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # --- Header ---
        c.setFillColorRGB(0, 0.33, 0.64)
        c.rect(0, height - 60, width, 60, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, height - 40, "UMKC Hospital Analytics")

        logo_path = os.path.join(BASE_DIR, "umkc_logo.png")
        if os.path.exists(logo_path):
            c.drawImage(logo_path, width - 140, height - 55, width=90, height=45, mask='auto')

        y = height - 100
        c.setFillColor(colors.HexColor("#0056A4"))
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "Patient Readmission Risk Report")
        y -= 20

        # --- Patient & Admission Details ---
                # --- Patient & Admission Details ---
        box_top = y
        box_height = 130
        c.setFillColor(colors.lightgrey)
        c.roundRect(35, box_top - box_height, width - 70, box_height, 10, fill=True, stroke=False)
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)

        # Left column (general info)
        c.drawString(50, box_top - 15, f"Patient Name: {data.get('Patient Name', 'N/A')}")
        c.drawString(50, box_top - 30, f"Admission Date: {data.get('Admission Date', 'N/A')}")
        c.drawString(50, box_top - 45, f"Simulation Date: {data.get('Simulation Date', 'N/A')}")
        c.drawString(50, box_top - 60, f"Age: {data.get('Age', 'N/A')} | Sex: {data.get('Sex', 'N/A')} | Weight: {data.get('Weight', 'N/A')} kg")
        c.drawString(50, box_top - 75, f"Blood Pressure: {data.get('Blood Pressure', 'N/A')} | Cholesterol: {data.get('Cholesterol', 'N/A')}")
        c.drawString(50, box_top - 90, f"Insulin: {data.get('Insulin', 'N/A')} | Diabetics Status: {data.get('Diabetics', 'N/A')}")

        # Right column (hospital + cardiac/diabetes markers)
        c.drawString(280, box_top - 15, f"Patient ID: {data.get('Patient ID', 'N/A')}")
        c.drawString(280, box_top - 30, f"Discharge Date: {data.get('Discharge Date', 'N/A')}")
        c.drawString(280, box_top - 45, f"Hospital / Unit: {data.get('Hospital Unit', 'N/A')}")

        if "heart" in disease.lower():
            c.setFont("Helvetica-Bold", 10)
            c.drawString(280, box_top - 65, "Cardiac Overview")
            c.setFont("Helvetica", 9)
            c.drawString(280, box_top - 80, f"ECG Result: {data.get('ECG Result', 'N/A')}")
            c.drawString(280, box_top - 95, f"Pulse Rate: {data.get('Pulse Rate (bpm)', 'N/A')} bpm")
        elif "diabet" in disease.lower():
            c.setFont("Helvetica-Bold", 10)
            c.drawString(280, box_top - 65, "Key Diabetes Markers")
            c.setFont("Helvetica", 9)
            c.drawString(280, box_top - 80, f"Hemoglobin: {data.get('Hemoglobin (g/dL)', 'N/A')}")
            c.drawString(280, box_top - 95, f"Urine Protein: {data.get('Urine Protein (mg/dL)', 'N/A')}")
            c.drawString(280, box_top - 110, f"Urine Glucose: {data.get('Urine Glucose (mg/dL)', 'N/A')}")

        y = box_top - box_height - 10  # move below gray box


        # --- Risk Summary ---
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(colors.HexColor("#0056A4"))
        c.drawString(40, y, "Risk Summary")
        y -= 18
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(50, y, f"Disease Type: {disease}")
        y -= 15
        c.drawString(50, y, f"Predicted Readmission: {'Yes' if adj_prob >= 0.5 else 'No'}")
        y -= 15
        c.drawString(50, y, f"Readmission Probability: {adj_prob:.4f} ({risk})")
        y -= 20

        # --- Clinical Visualization ---
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(colors.HexColor("#0056A4"))
        c.drawString(40, y, "Clinical Visualization")
        y -= 10

        prob = max(0.0, min(adj_prob, 1.0))
        d = Drawing(120, 100)
        pie = Pie()
        pie.x = 20
        pie.y = 10
        pie.width = 100
        pie.height = 100
        pie.data = [prob, 1 - prob]
        pie.labels = [f"Risk {prob:.2f}", f"Safe {1 - prob:.2f}"]
        pie.slices[0].fillColor = colors.red
        pie.slices[1].fillColor = colors.green
        d.add(pie)
        renderPDF.draw(d, c, 60, y - 110)

        chart_path = generate_signal_chart(disease, data)
        c.drawImage(ImageReader(chart_path), 250, y - 80, width=250, height=90)
        y -= 140

        # --- Follow-up Plan ---
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(colors.HexColor("#0056A4"))
        c.drawString(40, y, "Follow-up & Care Plan")
        y -= 20
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(50, y, f"Channel: {followup['channel']}")
        y -= 15
        c.drawString(50, y, f"Schedule: {', '.join(followup['schedule'])}")
        y -= 15
        c.drawString(50, y, f"Note: {followup['note']}")
        y -= 25

        # --- Staffing Suggestion ---
        c.setFont("Helvetica-Bold", 13)
        c.setFillColor(colors.HexColor("#0056A4"))
        c.drawString(40, y, "Resource Simulation Summary")
        y -= 20
        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(50, y, f"Expected Readmissions: {staffing['expected_readmissions']}")
        y -= 15
        c.drawString(50, y, f"Beds: {staffing['suggested_beds']} | Nurses: {staffing['suggested_nurses']} | Doctors: {staffing['suggested_doctors']}")
        y -= 50

        # --- Signature Line ---
        c.setStrokeColor(colors.black)
        c.line(width/2 - 100, y, width/2 + 100, y)
        c.setFont("Helvetica-Oblique", 10)
        c.drawCentredString(width / 2, y - 15, "Physician-in-Charge Signature")

        # --- Footer ---
        c.setFillColorRGB(0, 0.33, 0.64)
        c.rect(0, 0, width, 55, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2, 30, "© 2025 UMKC Hospital Analytics | AI-Driven Readmission Predictor")
        c.setFont("Helvetica", 8.5)
        c.drawCentredString(width / 2, 16, "UMKC Hospital Unit, Kansas City, Missouri, 64111")

        c.showPage()
        c.save()
        buffer.seek(0)
        # ================== PDF DESIGN END ==================

        # ================== PDF DESIGN END ==================


        return send_file(
            buffer,
            as_attachment=True,
            download_name="readmission_report.pdf",
            mimetype="application/pdf",
        )
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/followups", methods=["GET"])
def api_get_followups():
    try:
        df = pd.read_csv(FOLLOWUP_PATH)
        df["Prediction Date"] = pd.to_datetime(df["Prediction Date"], errors="coerce")
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=6)
        df_recent = df[(df["Prediction Date"] >= cutoff) & (df["Status"] != "Completed")]
        records = df_recent.sort_values("Prediction Date", ascending=False).to_dict(orient="records")
        return jsonify(records)
    except Exception as e:
        print(f"[ERROR] Could not fetch follow-ups: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/followup/complete", methods=["POST"])
def api_complete_followup():
    try:
        data = request.get_json()
        pid = data.get("Patient ID")

        if not pid:
            return jsonify({"error": "Patient ID required"}), 400

        df = pd.read_csv(FOLLOWUP_PATH)
        mask = df["Patient ID"].astype(str) == str(pid)
        if not mask.any():
            return jsonify({"error": f"No record found for Patient ID {pid}"}), 404

        df.loc[mask, "Status"] = "Completed"
        df.to_csv(FOLLOWUP_PATH, index=False)
        print(f"[INFO] Marked {pid} as Completed")

        return jsonify({"message": f"Patient {pid} marked as completed"})
    except Exception as e:
        print(f"[ERROR] Complete follow-up failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

