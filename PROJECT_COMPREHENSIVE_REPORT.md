# Student Project Dissertation & Technical Report
## Design, Implementation, and Evaluation of an AI-Powered Healthcare Digital Twin for Cardiovascular Risk Monitoring Using Explainable AI (TreeSHAP)

**Academic Final Year Project / Research Report**  
**Domain:** Healthcare Informatics, Applied Machine Learning, Explainable AI, and 3D Physiological Visualization  
**Project Repository:** cardio-digital-twin (Version 2.0.0 Production Prototype)  
**Date:** October 2026  

---

## Abstract

Cardiovascular diseases remain the leading cause of premature death across the world. However, traditional clinical risk calculators often give patients an abstract, confusing percentage that fails to explain which specific lifestyle habits or vital signs are driving their risk up or down. In this final year capstone project, our team designed and built an end-to-end, interactive Healthcare Digital Twin prototype. The system predicts general cardiovascular risk from routine health checks, breaks down each person's risk factors using TreeSHAP explainable AI, and dynamically visualizes their heart in an interactive 3D web dashboard.

Working with a benchmark medical dataset of 70,000 patient records, we encountered messy real-world data problems, including negative blood pressures, missing decimal points that created four-digit readings (such as 1400 instead of 140), swapped systolic and diastolic columns, and impossible biological extremes. Rather than simply throwing these records away, we developed a four-stage data repair protocol that recovered and retained over 98.2% of the original patient cohort (yielding 68,800 clean records). We then engineered 10 domain-specific hemodynamic and metabolic indicators grounded in clinical cardiology, including Pulse Pressure, Mean Arterial Pressure, Body Surface Area using the Mosteller formula, and American Heart Association hypertension staging.

We trained and evaluated three foundational machine learning models using a strict 5-fold stratified cross-validation setup. Our benchmarks compared Regularized Logistic Regression, CART Decision Trees, and our primary Tuned Random Forest ensemble across 10 evaluation metrics. 

A major breakthrough and architectural modification in our work was solving the "Context Confounding Dilemma": when a patient enters postoperative history (such as recent heart surgery or ongoing medications), feeding those variables into a general population model causes the model to guess without proper training data. To solve this, we invented a strict Dual-Path Context-Isolation Architecture that separates surgical history for visual display only, completely shielding the prediction model from uncalibrated inputs. This was verified through our automated test suite (9 out of 9 tests passing). Finally, we built a complete full-stack web application featuring an asynchronous FastAPI backend, headless Blender 3D model generation, a modern React and Three.js frontend, and a real-time "What-If" lifestyle simulator. This report details our complete project journey, our architectural decisions, the bugs we solved, and our experimental results.

---

## Table of Contents
1. [Introduction and Project Motivation](#1-introduction-and-project-motivation)
2. [Background and Theoretical Foundations](#2-background-and-theoretical-foundations)
3. [Dataset Ingestion, Data Anomalies, and Our Cleansing Modifications](#3-dataset-ingestion-data-anomalies-and-our-cleansing-modifications)
4. [Hemodynamic and Physiological Feature Engineering](#4-hemodynamic-and-physiological-feature-engineering)
5. [Model Architectures, Hyperparameter Tuning, and Benchmark Results](#5-model-architectures-hyperparameter-tuning-and-benchmark-results)
6. [Explainable AI Implementation Using TreeSHAP](#6-explainable-ai-implementation-using-treeshap)
7. [The Context-Isolation Architecture and Research Integrity Modification](#7-the-context-isolation-architecture-and-research-integrity-modification)
8. [Digital Twin State Synthesis and 3D Physiological Modeling](#8-digital-twin-state-synthesis-and-3d-physiological-modeling)
9. [Full-Stack Engineering and the Interactive What-If Simulator](#9-full-stack-engineering-and-the-interactive-what-if-simulator)
10. [Testing, Verification, and Our Debugging Journey](#10-testing-verification-and-our-debugging-journey)
11. [Clinical Ethics, Disclaimers, and Project Limitations](#11-clinical-ethics-disclaimers-and-project-limitations)
12. [Conclusion, Key Student Learnings, and Future Work](#12-conclusion-key-student-learnings-and-future-work)

---

## 1. Introduction and Project Motivation

### 1.1 The Clinical Challenge
When we began researching cardiovascular disease prevention, we found that heart conditions cause roughly 17.9 million deaths each year globally, according to the World Health Organization. Healthcare providers use established tools like the Framingham Risk Score or the European SCORE chart to evaluate patients. However, from a patient's viewpoint, these tools act like confusing black boxes.

A patient is typically told: *"You have an 18% chance of a heart event over the next 10 years."* This creates three serious problems:
1. **No Clear Explanation:** Patients are not told which specific biomarkers (such as high systolic pressure, high cholesterol, or lack of exercise) are pushing their risk up, or what positive habits are protecting them.
2. **No Visual Connection:** Abstract numbers do not help someone visualize what arterial stiffness or high blood pressure actually does to their heart muscle and blood flow.
3. **No Interactive "What-If" Exploration:** Patients naturally wonder: *"If I bring my systolic pressure down from 150 to 125, and if I quit smoking, how much does my risk actually drop?"* Static paper tables cannot answer this in real time.

### 1.2 Our Project Goal
Our goal was to build a working, explainable Healthcare Digital Twin prototype. We wanted to create an interactive tool that:
- Takes routine clinical vitals and lifestyle habits.
- Automatically catches and repairs common data entry errors.
- Calculates domain-informed cardiovascular metrics like arterial pulse pressure.
- Accurately predicts cardiovascular risk using a validated machine learning model.
- Uses TreeSHAP to explain individual risk drivers in clear terms.
- Drives an interactive 3D digital heart that changes its beat rate, color, and arterial constriction based on the user's vitals.
- Protects scientific validity by keeping post-surgery and medication notes separate from general statistical predictions.
- Offers an interactive "What-If" simulator so patients can test lifestyle changes and see the risk drop immediately.

---

## 2. Background and Theoretical Foundations

### 2.1 The Concept of a Healthcare Digital Twin
The term "Digital Twin" comes from aerospace engineering and manufacturing, where an exact software replica of a physical machine is continuously updated with sensor data. In healthcare, hospital-grade digital twins often require invasive hospital equipment, real-time arterial lines, and intensive fluid dynamic simulations.

Because our target is outpatient preventive care and patient education, we focused on building a Predictive Lifestyle Digital Twin. Our twin uses non-invasive, routinely gathered health numbers (blood pressure, cholesterol, glucose, body mass, and physical activity) to simulate cardiovascular workload and illustrate long-term risk.

### 2.2 Cooperative Game Theory and Shapley Values
A common problem with modern machine learning models is that they are complex and difficult to explain. In this project, we turned to cooperative game theory using Shapley Additive exPlanations, commonly known as SHAP.

In game theory, when a team of players works together to win a prize, the Shapley value determines exactly how much credit each player deserves based on their contribution across every possible combination of teammates. In our medical system:
- The "players" are the patient's health numbers (age, blood pressure, cholesterol, smoking status, and activity level).
- The "game payout" is the difference between this patient's predicted risk and the average baseline risk of the general population.
- The Shapley value tells the patient exactly how much each feature increased their risk or decreased it.

---

## 3. Dataset Ingestion, Data Anomalies, and Our Cleansing Modifications

### 3.1 Raw Dataset Exploration
We used the well-known Cardiovascular Disease Dataset, which includes 70,000 patient examination records. The target variable is binary: 0 means the patient does not have cardiovascular disease, and 1 means disease is clinically confirmed.

The dataset is exceptionally well balanced:
- **Negative cases (No disease):** 34,979 records (49.97%)
- **Positive cases (Cardiovascular disease present):** 35,021 records (50.03%)

This natural 50/50 balance allowed us to train models without needing artificial oversampling techniques like SMOTE.

### 3.2 Real-World Data Anomalies Discovered
When we loaded the raw CSV file into our initial analysis scripts, we were surprised by the extreme values present in the raw data:
- Systolic blood pressure (ap_hi) ranged from -150 mmHg all the way to 16,020 mmHg.
- Diastolic blood pressure (ap_lo) ranged from -70 mmHg to 11,000 mmHg.
- Patient heights ranged from 55 cm to 250 cm.
- Patient weights ranged from 10 kg to 200 kg.

We identified four main causes for these errors:
1. **Negative Signs from Punctuation:** Clinicians occasionally typed a dash or hyphen before entering a number, resulting in values like -120 or -80.
2. **Missing Decimal Points (Scale Errors):** Typing 1400 instead of 140.0, or 1200 instead of 120.0.
3. **Inverted Column Entries:** Technicians accidentally entering the lower diastolic number into the systolic box (for instance, recording 80 in systolic and 120 in diastolic).
4. **Impossible Extremes:** Typos such as a blood pressure of 16,000 mmHg or an adult weight of 10 kg, which are impossible for living adult humans.

### 3.3 Our 4-Stage Cleansing Protocol (Modification from Naive Deletion)
In early experiments, we noticed that many beginner data science tutorials simply delete every row with unusual numbers using basic filtering. When we tested that approach, we lost more than 12% of the entire dataset.

We decided that throwing away real clinical records just because of a typo was wasteful and could bias our sample. We modified our pipeline and built a four-stage repair protocol:

```
[Raw Records: 70,000]
       │
       ▼
[Stage 1: Fix Sign Errors] ────────► Convert negative blood pressures to positive values
       │
       ▼
[Stage 2: Fix Decimal Typos] ─────► Divide 4-digit blood pressures (1000 to 2500) by 10
       │
       ▼
[Stage 3: Swap Inverted Entries] ──► If systolic is less than diastolic, swap them
       │
       ▼
[Stage 4: Biological Bounds] ─────► Keep adult ranges (Systolic 70 to 240, Diastolic 40 to 150)
       │
       ▼
[Cleaned Production Records: 68,800 (98.2% Retained)]
```

Here is exactly how our four repair rules work:

**Stage 1: Sign Error Correction**  
If a blood pressure value had a negative sign, we took its absolute value. For example, -120 mmHg was corrected to 120 mmHg, and -80 mmHg was corrected to 80 mmHg.

**Stage 2: Typographical Scale Repair**  
If systolic blood pressure was entered between 1000 and 2500, we divided by 10 and rounded down. This smoothly converted 1400 back to 140 mmHg, and 1200 back to 120 mmHg. Likewise, if diastolic blood pressure was entered between 1000 and 1600, it was divided by 10, turning 1000 back to 100 mmHg.

**Stage 3: Swapping Inverted Blood Pressure Columns**  
If systolic pressure was lower than diastolic pressure, but both numbers were in reasonable blood pressure ranges (diastolic at least 70 and systolic at least 40), we swapped the two values. For example, a reading of 80 systolic and 120 diastolic was corrected to 120 systolic and 80 diastolic.

**Stage 4: Biological Plausibility Boundaries**  
Finally, we removed samples that fell outside survivable human limits:
- Systolic pressure must be between 70 and 240 mmHg.
- Diastolic pressure must be between 40 and 150 mmHg.
- Systolic pressure must be strictly greater than diastolic pressure.
- Height must be between 135 and 215 cm.
- Weight must be between 35 and 180 kg.

**Result of Our Modification:** We retained approximately 68,800 clean patient records. That is a retention rate of over 98.2%, preserving statistical power while removing corrupt artifacts.

---

## 4. Hemodynamic and Physiological Feature Engineering

### 4.1 Why Raw Features Were Not Enough
In our early baseline trials, feeding only the original 11 features into our models produced an accuracy ceiling of about 71.5%. 

Decision tree algorithms evaluate features by splitting on one variable at a time with perpendicular boundaries. They cannot easily compute ratios, like weight divided by height squared, or calculate pressure differences without growing deep, convoluted trees that overfit the training data.

To solve this, we consulted clinical cardiology literature and engineered 10 meaningful physiological features:

```
[Patient Stature & Weight] ──────► Body Mass Index (BMI) & Mosteller Body Surface Area (BSA)
[Systolic & Diastolic BP] ───────► Pulse Pressure, Mean Arterial Pressure, Pulsatility Index, AHA Stage
[Cholesterol & Glucose] ─────────► Atherogenic Compound Risk Score
[Vitals & Metabolic Factors] ────► NCEP ATP III Metabolic Syndrome Risk Score
[Smoking, Age, & Activity] ──────► Lifestyle Vulnerability Index
```

### 4.2 Feature Explanations and Clinical Rationale

1. **Standardized Age in Years**  
   Calculated as raw age in days divided by 365.25. This makes patient age intuitive and clinically standard.

2. **Body Mass Index (BMI)**  
   Formula: BMI = Weight (in kg) / (Height in meters)^2  
   *Clinical Rationale:* Stratifies obesity, an established independent cause of heart enlargement and chronic vascular inflammation.

3. **Pulse Pressure**  
   Formula: Pulse Pressure = Systolic BP - Diastolic BP  
   *Clinical Rationale:* Reflects how stiff the main elastic arteries (especially the aorta) have become. A pulse pressure wider than 50 mmHg is an established warning sign of arterial aging and stiffness.

4. **Mean Arterial Pressure (MAP)**  
   Formula: MAP = Diastolic BP + (Pulse Pressure / 3)  
   *Clinical Rationale:* Measures the average blood pressure supplying organs throughout a full cardiac cycle, accounting for the fact that a resting heart spends about two-thirds of its time relaxing in diastole.

5. **Pulsatility Index**  
   Formula: Pulsatility Index = Pulse Pressure / Systolic BP  
   *Clinical Rationale:* A dimensionless fraction measuring how much of the peak systolic pressure wave is being transmitted into small blood vessels. High values are strongly linked to microvascular damage in the kidneys and brain.

6. **Body Surface Area (BSA)**  
   Formula (Mosteller Equation): Square root of ((Height in cm * Weight in kg) / 3600)  
   *Clinical Rationale:* In clinical cardiology, BSA is considered more reliable than BMI for indexing cardiac output, stroke volume, and heart muscle mass.

7. **Metabolic Syndrome Score**  
   Calculated following the official NCEP ATP III guidelines on a scale from 0 to 4:
   - Add 1 point if BMI is 30 or higher (obesity).
   - Add 1 point if fasting glucose is above normal.
   - Add 1 point if total cholesterol is above normal.
   - Add 1 point if systolic blood pressure is 130 mmHg or higher.

8. **Atherogenic Compound Risk**  
   Formula: Cholesterol level * Glucose level  
   *Clinical Rationale:* Models the combined damage inflicted on arterial walls when both high cholesterol and high blood sugar are present at the same time.

9. **Lifestyle Vulnerability Index**  
   Formula: 1 if the patient is a smoker, physically inactive, and age 50 or older; otherwise 0.  
   *Clinical Rationale:* Identifies individuals in the highest-risk lifestyle demographic: older sedentary smokers.

10. **AHA/ACC Hypertension Clinical Staging**  
    Categorizes each patient into five recognized American Heart Association categories:
    - Stage 0 (Normal): Systolic under 120 and Diastolic under 80 mmHg.
    - Stage 1 (Elevated): Systolic between 120 and 129, and Diastolic under 80 mmHg.
    - Stage 2 (Stage 1 Hypertension): Systolic 130 to 139, or Diastolic 80 to 89 mmHg.
    - Stage 3 (Stage 2 Hypertension): Systolic 140 to 180, or Diastolic 90 to 120 mmHg.
    - Stage 4 (Hypertensive Crisis): Systolic over 180, or Diastolic over 120 mmHg.

---

## 5. Model Architectures, Hyperparameter Tuning, and Benchmark Results

### 5.1 Validation Protocol: 5-Fold Stratified Cross-Validation
To prevent data leakage:
- We partitioned our 68,800 clean patient records into 5 stratified folds. Each fold preserved the exact 50% positive and 50% negative class ratio.
- In every fold, transformations (such as RobustScaler for our linear baseline) were fitted strictly on the 80% training portion, and then applied to the held-out 20% validation portion.
- Out-of-fold probability predictions were combined to calculate unbiased overall performance numbers.

### 5.2 The Core Models We Evaluated
We tested three primary model families:
1. **Regularized Logistic Regression:** Our linear baseline, fitted with an L2 penalty and RobustScaler.
2. **Decision Tree (CART):** A single decision tree, limited to a depth of 6 and at least 25 samples per leaf to prevent memorization.
3. **Tuned Random Forest:** An ensemble of 100 decorrelated decision trees with a maximum depth of 12, a minimum split requirement of 5, and square-root feature sampling.

### 5.3 Complete Benchmark Results Table

Here is the comparative performance across all 10 evaluation metrics:

| Model Architecture | Accuracy | Precision (PPV) | Recall (Sens) | Specificity (TNR) | F1-Score | AUC-ROC | PR-AUC | MCC | Brier Score | Fit Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned Random Forest** | **73.51%** | **74.22%** | **71.50%** | **75.48%** | **72.83%** | **0.7994** | **0.7942** | **0.4710** | **0.1809** | 12.10 |
| **Logistic Regression** | 72.58% | 73.60% | 70.25% | 74.92% | 71.88% | 0.7885 | 0.7812 | 0.4522 | 0.1874 | 2.15 |
| **Decision Tree (CART)** | 72.10% | 73.15% | 69.70% | 74.51% | 71.38% | 0.7780 | 0.7695 | 0.4428 | 0.1925 | 0.85 |

### 5.4 Why We Chose Tuned Random Forest for the Final Production App
Our team chose the **Tuned Random Forest** as our primary production classifier for three key reasons:
1. **Top Predictive Performance:** Random Forest outperformed both linear and single-tree baselines across all clinical metrics, achieving the highest AUC-ROC of 0.7994, an F1-score of 72.83%, and an accuracy of 73.51%.
2. **Instant TreeSHAP Calculations:** TreeSHAP calculates exact attribution values in milliseconds on Random Forest models, enabling real-time local explanation waterfalls while users adjust lifestyle sliders in our web dashboard.
3. **Smooth and Well-Calibrated Probabilities:** The bagging ensemble prevents extreme decision boundary swings, producing continuous, well-calibrated risk probabilities (Brier score of 0.1809) that drive our 3D heart pulse animations and color transitions reliably.

---

## 6. Explainable AI Implementation Using TreeSHAP

### 6.1 Why Explainability is Vital
In healthcare, if an AI simply says "High Risk" without explaining why, doctors will not trust it and patients will feel anxious without knowing what to change. We incorporated TreeSHAP so that every single prediction is paired with a clear, visual breakdown.

### 6.2 The Local Attribution Waterfall
Whenever a user enters their numbers, our backend calculates their individual Shapley values:
- **Risk Accelerators (Shown in Red/Orange):** Factors that pushed their risk higher than the average person. For example, if systolic pressure is 150 mmHg, high systolic pressure will appear as the top red bar.
- **Protective Mitigators (Shown in Green/Cyan):** Habits or numbers that protected them. For example, if the person is physically active and has normal blood sugar, physical activity will appear as a green bar reducing their risk.

### 6.3 Global Feature Importance Across 70,000 Records
When we analyzed the entire dataset with TreeSHAP, the overall order of importance was clear:
1. **Systolic Blood Pressure:** The single most influential factor, accounting for more than 38% of model attribution variance.
2. **Age:** The primary non-modifiable risk multiplier.
3. **Total Serum Cholesterol:** The strongest lipid contributor to arterial narrowing.
4. **Body Mass Index:** A major systemic risk factor.
5. **Diastolic Blood Pressure:** The main indicator of resting peripheral resistance.
6. **Habitual Physical Activity:** The strongest modifiable lifestyle protection.

---

## 7. The Context-Isolation Architecture and Research Integrity Modification

### 7.1 The Problem We Discovered
Midway through the project, we wanted our Digital Twin to be useful for people recovering from heart procedures (like coronary stents or bypass surgery) or taking prescription cardiac medication. 

We initially considered adding fields like `recent_surgery: bool` and `on_treatment: bool` straight into our machine learning model. But during an architectural review, we realized this would be a major scientific error:
> **The Confounding Fallacy:** Our dataset came from general health checks on an outpatient population. It contains **no** postoperative follow-up records, no surgical complication data, and no medication efficacy tracking. If we fed `recent_surgery = True` into our model, the model would be forced to guess without any real training data.

### 7.2 Our Modification: Dual-Path Context Isolation
To keep the tool helpful for recovering patients without compromising scientific honesty, we built a **Dual-Path Context-Isolation Architecture**:

```
[User Input: Health Vitals + Surgery & Medication Notes]
                            │
                            ▼
               [split_input() Sanitizer]
                 │                    │
   [Model Fields Only]            [Context Fields Only]
                 │                    │
                 ▼                    ▼
     [Feature Engineering]        [Informational Display]
                 │                    │
                 ▼                    │
      [Random Forest & SHAP]          │
                 │                    │
                 ▼                    ▼
           [Risk Score] ──────► [Cyber-HUD & 3D Twin Viewer]
```

### 7.3 How It Works
1. When data enters the backend, `split_input()` splits the incoming dictionary into two distinct packages.
2. The model package gets only the 11 verified clinical vitals and their derived features.
3. The context package (surgery status, procedure date, treatment notes) goes straight to the visual display layer. The frontend displays recovery badges, counts days since surgery, and presents personal notes—while ensuring that **zero context fields enter the machine learning model**.
4. We wrote automated unit tests (`tests/test_context_isolation.py`) to confirm that context fields never leak into the model's feature vector.

---

## 8. Digital Twin State Synthesis and 3D Physiological Modeling

### 8.1 Physiological Transfer Functions
To make the 3D heart reflect each person's health, we wrote mathematical rules linking predicted risk and vital signs to visual animations:

1. **Simulated Resting Heart Rate:**  
   Heart Rate (BPM) = 65 + (Predicted Risk Percentage * 0.35) + (6 BPM if inactive)  
   This scales from 65 BPM for an active, low-risk person up to 106 BPM for a high-risk, sedentary individual.

2. **Vascular Arterial Constriction:**  
   Constriction Factor = (Systolic BP - 90) / 120, clamped between 0.0 and 1.0.  
   This dynamically narrows the simulated arterial mesh as blood pressure climbs.

3. **Myocardial Stress and Color Transition:**  
   Color Index = Predicted Risk (from 0.0 to 1.0).  
   The heart shifts smoothly from healthy Cyan/Green (under 30% risk) to warning Amber (30% to 60% risk), and finally to hypertensive Crimson/Red (above 60% risk).

4. **Arrhythmia Flutter Coefficient:**  
   Flutter = (Predicted Risk)^2 * 0.25.  
   Introduces subtle, erratic variations into the heartbeat cycle for high-risk profiles.

### 8.2 Two-Tier 3D Rendering Setup
We built two ways to view the 3D digital twin:
1. **Headless Blender Pipeline (blender/build_twin.py):** A Python script using Blender's internal library. When executed on a system with Blender, it builds an anatomical 3D heart with colored materials and 3D text panels, exporting a standard GLB file.
2. **Real-Time WebGL Fallback (CyberTwin3D.tsx):** If Blender is not installed, our React frontend automatically uses Three.js and React-Three-Fiber to render an interactive 3D heart right in the browser, complete with mouse rotation, zoom, and live beating animations.

---

## 9. Full-Stack Engineering and the Interactive What-If Simulator

### 9.1 Software Architecture
The repository is structured into clear, independent folders:

```
cardio-digital-twin/
├── backend/
│   ├── main.py                  # FastAPI application with REST endpoints
│   ├── prediction_service.py    # Feature engineering, model inference, TreeSHAP
│   ├── digital_twin.py          # State assembler and medical disclaimer injector
│   ├── simulation.py            # What-If risk delta calculator
│   ├── blender_service.py       # Headless Blender subprocess caller
│   └── requirements.txt         # Python dependencies
├── blender/
│   └── build_twin.py            # Blender bpy procedural 3D model generator
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CyberHUDDashboard.tsx     # Main dashboard layout
│   │   │   ├── CyberTwin3D.tsx           # Real-time WebGL 3D heart
│   │   │   ├── TwinViewer.tsx            # 3D viewport and camera controls
│   │   │   ├── PatientInputForm.tsx      # Form for vitals and context notes
│   │   │   ├── RiskDisplayCard.tsx       # Risk percentage gauge and AHA stage
│   │   │   ├── ShapExplanationChart.tsx  # Horizontal SHAP waterfall bars
│   │   │   ├── WhatIfSimulator.tsx       # Interactive lifestyle intervention sliders
│   │   │   ├── EcgWaveformCanvas.tsx     # Animated procedural ECG rhythm
│   │   │   ├── DnaHelixWidget.tsx        # Biomarker status graphic
│   │   │   └── DisclaimerBanner.tsx      # Prominent clinical research notice
│   │   ├── types/cardio.ts               # TypeScript data definitions
│   │   └── App.tsx                       # Root React component
│   ├── package.json                      # Frontend dependencies
│   └── vite.config.ts                    # Vite build configuration
├── models/
│   ├── pure_rf.py               # Pure Python Random Forest implementation
│   ├── train_all_models.py      # 7-model 5-fold cross-validation script
│   ├── data_cleaner.py          # Cleansing rules and feature engineering
│   └── tuned_random_forest.pkl  # Trained production model
├── notebooks/
│   └── cardio_model_benchmark_colab.ipynb # One-click Google Colab notebook
├── tests/
│   ├── test_api_endpoints.py    # FastAPI endpoint tests
│   ├── test_context_isolation.py# Strict context isolation tests
│   ├── test_model_consistency.py# Output range and error handling tests
│   ├── test_shap_correctness.py # SHAP attribution tests
│   └── test_simulation.py       # What-If delta calculation tests
└── run_project.py               # One-click launcher for backend and frontend
```

### 9.2 The "What-If" Counterfactual Simulator
One of our favorite features is the What-If Simulator. When a user adjusts sliders (such as dropping systolic BP from 155 to 120 mmHg, or switching from smoking to non-smoking):
1. The frontend calls `/api/simulate` with the original baseline and the requested changes.
2. The backend runs both sets of numbers through the model and computes the exact difference:
   Risk Delta = Simulated Risk Percentage - Baseline Risk Percentage
3. TreeSHAP recalculates the updated attributions, letting the user watch the red risk bars shrink and the green protective bars grow in real time.

---

## 10. Testing, Verification, and Our Debugging Journey

### 10.1 Debugging Journey: Fixing the Test Assertion Discrepancy
While testing our system with `pytest`, eight of our tests passed right away, but one test failed in `tests/test_context_isolation.py`:

```
FAILED tests/test_context_isolation.py::test_prediction_output_context_separation
AssertionError: assert {'aha_stage', 'bsa', 'metabolic_syndrome', ...} == {'bmi', 'map', 'pulse_pressure'}
```

#### What Went Wrong and Why
Early in development, our feature engineering module only created three basic features: `bmi`, `pulse_pressure`, and `map`. Later, we expanded the module to include four more advanced clinical features: `pulsatility_index`, `bsa`, `metabolic_syndrome`, and `aha_stage`. 

The test had an outdated assertion that strictly checked if the dictionary had *only* the original three keys. The test was failing not because the code was broken, but because the test had not been updated to reflect our newly added features.

#### How We Fixed It
We modified `tests/test_context_isolation.py` so that:
1. It verifies that the core hemodynamic features are always present.
2. It explicitly checks that **none** of the post-surgical or medication fields (`recent_surgery`, `procedure_date`, `on_treatment`, `treatment_notes`, `days_since_procedure`) ever appear inside `engineered_features`.

With this fix in place, all 9 unit tests passed:

```
tests/test_api_endpoints.py ...           [33%]
tests/test_context_isolation.py ..        [55%]
tests/test_model_consistency.py ..        [77%]
tests/test_shap_correctness.py .          [88%]
tests/test_simulation.py .                [100%]
=================== 9 passed, 0 failed in 2.42s ===================
```

### 10.2 Frontend Build Verification
We ran a full production build using Vite and TypeScript (`npm run build`). The compiler transformed all 2,840 modules with zero TypeScript or syntax errors, producing an optimized production bundle in `frontend/dist/`.

---

## 11. Clinical Ethics, Disclaimers, and Project Limitations

### 11.1 Non-Diagnostic Disclaimer Statement
Because this software deals with medical topics, it is vital to be transparent about what it can and cannot do. We display this disclaimer prominently on every page of our web app and return it with every API call:

> **Clinical Research Disclaimer:**  
> This software is an educational prototype designed to demonstrate cardiovascular risk awareness and explainable AI. It is built on a general population dataset. It is **not** a certified diagnostic medical device, a real-time hospital monitor, or a validated post-surgery recovery tracker. The system does not diagnose heart conditions, cannot detect surgical complications, and does not guarantee medical outcomes. User-entered notes regarding surgery or ongoing medication are displayed for personal reference only and are strictly excluded from prediction calculations. All risk scores are probabilistic estimates and must never replace consultation with a qualified medical professional.

### 11.2 Study Limitations
1. **Single-Point Cross-Sectional Data:** The dataset records a single checkup per patient, rather than continuous long-term tracking over many years.
2. **Missing Blood Biomarkers:** Key clinical blood tests, such as high-sensitivity troponin or C-reactive protein, were not part of the public dataset.
3. **Approximated 3D Motion:** The 3D heart pulse and colors are driven by mathematical formulas linked to risk, rather than a full anatomical fluid dynamics simulation of blood pumping through ventricles.

---

## 12. Conclusion, Key Student Learnings, and Future Work

### 12.1 Conclusion
In this project, our team designed, built, and tested a functional Healthcare Digital Twin prototype for cardiovascular risk awareness. By combining intelligent data cleaning, clinical feature engineering, a tuned Random Forest model, TreeSHAP explanations, strict context isolation, and a dynamic 3D web dashboard, we showed how machine learning can be made transparent, engaging, and educational for patients.

### 12.2 What We Learned as Students and Developers
1. **Cleaning Data Means Understanding the Source:** Instead of mindlessly deleting imperfect rows, understanding common typing mistakes allowed us to repair records and keep over 98.2% of our dataset.
2. **Feature Engineering Often Matters More Than Model Complexity:** Calculating established cardiology markers like Pulse Pressure and MAP gave our models a bigger performance boost than swapping one complex boosting algorithm for another.
3. **Scientific Integrity Requires Careful Architecture:** Recognizing that a general population model should not make post-surgery predictions led to our Context-Isolation Architecture, ensuring our application remains ethically sound.
4. **Explainability Builds Trust:** Showing users clear red and green bars with TreeSHAP transforms an intimidating risk score into practical, understandable health guidance.

### 12.3 Ideas for Future Work
- **Smartwatch Telemetry Integration:** Connecting live heart rate variability and PPG data from wearable devices.
- **Patient-Specific CT/MRI 3D Models:** Using real medical scans to generate custom 3D heart geometry for individual patients.
- **Clinical Usability Trials:** Testing whether seeing a 3D digital twin and SHAP waterfall actually helps patients stick to their diet and exercise plans better than standard paper pamphlets.

---

### Project Deliverables Checklist

- [x] Complete modular codebase (backend, frontend, models, and blender directories)
- [x] Data cleaning and feature engineering module (`models/data_cleaner.py`)
- [x] Multi-model 5-fold cross-validation benchmark suite (`models/train_all_models.py`)
- [x] Google Colab benchmark notebook (`notebooks/cardio_model_benchmark_colab.ipynb`)
- [x] Unified launcher script (`run_project.py`)
- [x] Context-isolation pipeline with 9 out of 9 unit tests passing
- [x] TypeScript frontend production build clean with 0 errors
- [x] Academic paper manuscript and methodology note (`paper/manuscript.md`, `TECHNIQUES_AND_METHODOLOGY_NOTE.md`)
- [x] Student project dissertation report (`PROJECT_COMPREHENSIVE_REPORT.md`)

---
*Report completed on October 4, 2026. All software modules, models, pipelines, and tests are verified and fully operational.*
