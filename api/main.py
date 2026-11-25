
from pathlib import Path
from typing import Literal
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "saved" /
    "best_legal_autoencoder.keras"
)

PREPROCESSOR_PATH = (
    BASE_DIR / "dataset" / "processed" /
    "legal_model_ready" / "legal_preprocessor.joblib"
)

THRESHOLD_PATH = (
    BASE_DIR / "reports" /
    "legal_anomaly_threshold.csv"
)


# ==========================================
# LOAD TRAINED COMPONENTS
# ==========================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

preprocessor = joblib.load(PREPROCESSOR_PATH)

scaler = preprocessor["scaler"]
encoder = preprocessor["encoder"]

threshold_data = pd.read_csv(THRESHOLD_PATH)

ANOMALY_THRESHOLD = float(
    threshold_data.loc[0, "AnomalyThreshold"]
)


# ==========================================
# CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="BillTimeGuard AI",
    description="AI-based legal billing anomaly detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# BILLING ENTRY INPUT FORMAT
# ==========================================

class BillingEntry(BaseModel):

    EntryDurationHours: float = Field(gt=0)
    Rate: float = Field(gt=0)

    DayOfWeek: int = Field(ge=0, le=6)
    Month: int = Field(ge=1, le=12)

    Partner: int = Field(ge=0, le=1)

    Position: Literal[
        "Associate",
        "Attorney",
        "Construction Attorney",
        "E-Discovery Attorney",
        "Litigation Counsel",
        "Litigation Technology Services Analyst",
        "Paralegal",
        "Partner",
        "Senior Associate",
        "Technical Specialist",
        "Unknown"
    ]


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to BillTimeGuard AI",
        "status": "API is running"
    }


# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ==========================================
# AI BILLING ANOMALY PREDICTION
# ==========================================

@app.post("/predict")
def predict(entry: BillingEntry):

    try:

        # Convert input into a DataFrame
        input_df = pd.DataFrame([entry.model_dump()])

        # Separate numerical and categorical features
        numeric_features = preprocessor["numeric_features"]

        categorical_features = preprocessor["categorical_features"]

        # Apply saved preprocessing
        numeric_data = scaler.transform(
            input_df[numeric_features]
        )

        categorical_data = encoder.transform(
            input_df[categorical_features]
        )

        # Combine into 16 model features
        model_input = np.hstack([
            numeric_data,
            categorical_data
        ]).astype(np.float32)

        # Reconstruct input using Autoencoder
        reconstructed = model.predict(
            model_input,
            verbose=0
        )

        # Calculate reconstruction error
        reconstruction_error = float(
            np.mean(
                np.square(model_input - reconstructed)
            )
        )

        # Compare against saved threshold
        is_anomaly = (
            reconstruction_error > ANOMALY_THRESHOLD
        )

        return {
            "anomaly_score": reconstruction_error,

            "threshold": ANOMALY_THRESHOLD,

            "status": (
                "Potential Anomaly"
                if is_anomaly else "Normal Range"
            ),

            "message": (
                "This entry differs from learned billing patterns."
                if is_anomaly else
                "This entry is within the model's learned range."
            ),

            "disclaimer": (
                "An anomaly flag is an investigation signal, "
                "not proof of fraud or misconduct."
            )
        }

    except Exception as e:

        print(f"Prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction."
        )


# ==========================================
# PDF REPORT INPUT FORMAT
# ==========================================

class ReportRequest(BaseModel):

    EntryDurationHours: float
    Rate: float

    DayOfWeek: int
    Month: int

    Partner: int
    Position: str

    anomaly_score: float
    threshold: float

    status: str
    message: str
    disclaimer: str

    review_indicators: list[str]


# ==========================================
# GENERATE PDF ANALYSIS REPORT
# ==========================================

@app.post("/generate-report")
def generate_report(report: ReportRequest):

    try:

        # Create PDF in memory
        buffer = BytesIO()

        pdf = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        content = []

        # Report title
        content.append(
            Paragraph(
                "BillTimeGuard AI",
                styles["Title"]
            )
        )

        content.append(
            Paragraph(
                "Legal Billing Anomaly Analysis Report",
                styles["Heading2"]
            )
        )

        content.append(Spacer(1, 20))

        # Billing entry details
        content.append(
            Paragraph(
                "Billing Entry Details",
                styles["Heading2"]
            )
        )

        details = [
            ["Field", "Value"],

            [
                "Duration (hours)",
                str(report.EntryDurationHours)
            ],

            [
                "Hourly Rate",
                f"${report.Rate:.2f}"
            ],

            [
                "Calculated Billed Amount",
                f"${report.EntryDurationHours * report.Rate:.2f}"
            ],

            [
                "Day of Week",
                str(report.DayOfWeek)
            ],

            [
                "Month",
                str(report.Month)
            ],

            [
                "Partner",
                "Yes" if report.Partner == 1 else "No"
            ],

            [
                "Professional Position",
                report.Position
            ]
        ]

        table = Table(
            details,
            colWidths=[220, 250]
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1B2B4B")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),
            ])
        )

        content.append(table)

        content.append(Spacer(1, 20))

        # AI analysis result
        content.append(
            Paragraph(
                "AI Analysis Result",
                styles["Heading2"]
            )
        )

        result_details = [
            ["Metric", "Value"],

            ["Status", report.status],

            [
                "Anomaly Score",
                f"{report.anomaly_score:.8f}"
            ],

            [
                "Detection Threshold",
                f"{report.threshold:.8f}"
            ]
        ]

        result_table = Table(
            result_details,
            colWidths=[220, 250]
        )

        result_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1B2B4B")
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    9
                ),
            ])
        )

        content.append(result_table)

        content.append(Spacer(1, 15))

        # Review indicators
        content.append(
            Paragraph(
                "Review Indicators",
                styles["Heading2"]
            )
        )

        for indicator in report.review_indicators:

            content.append(
                Paragraph(
                    f"- {indicator}",
                    styles["BodyText"]
                )
            )

            content.append(Spacer(1, 5))

        content.append(Spacer(1, 15))

        # Analysis message
        content.append(
            Paragraph(
                report.message,
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 10))

        # Disclaimer
        content.append(
            Paragraph(
                report.disclaimer,
                styles["Italic"]
            )
        )

        # Build PDF
        pdf.build(content)

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                "attachment; filename=BillTimeGuard_Analysis_Report.pdf"
            }
        )

    except Exception as e:

        print(f"PDF generation error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate the PDF report."
        )
