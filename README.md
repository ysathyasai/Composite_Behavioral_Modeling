<div align="center">
  <h1>Composite Behavioral Model</h1>
  <img src="https://img.shields.io/badge/Model-Composite--Behavioral-0052cc?style=flat-square&logo=diagrams.net" />
  <br />
  <a href="https://www.djangoproject.com/"><img src="https://img.shields.io/badge/Django-3.2-092E20?style=flat-square&logo=django&logoColor=white" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" /></a>
  
  [🚀 Live Demo](https://composite-behavioral-modeling-abci.onrender.com/) • [📖 Documentation](#-features) • [📄 LICENSE](#-license)
</div>

---

## 🚀 Project Overview

`ysathyasai` is a Django-based identity theft detection platform that evaluates fraud risk using behavioral and transactional inputs. The system provides:

- A remote user portal for registration, login, and prediction
- A service provider admin portal for ML training and analytics
- A statistical dashboard for model performance and fraud distribution
- Exportable prediction results for offline review

**This is my 6th semester project** - a comprehensive implementation of machine learning in web development for fraud detection.

---

## ✨ Features

- **Remote User Experience**
  - Registration and login flow
  - Profile dashboard
  - Real-time fraud prediction entry
- **Service Provider Dashboard**
  - Admin authentication
  - Model training for multiple classifiers
  - Performance accuracy metrics
  - Downloadable prediction exports
- **Statistical Reporting**
  - Model accuracy comparisons
  - Fraud vs no-fraud distribution
  - Trend-ready analytics view

---

## 📁 Repository Structure

```text
ysathyasai/
├── composite_behavioral_modeling/
│   ├── composite_behavioral_modeling/  # Django settings and project config
│   ├── Remote_User/                  # Remote user models and views
│   └── Service_Provider/             # Admin views, ML training, reports
├── Template/htmls/                   # UI templates and static HTML pages
├── Datasets.csv                      # Dataset used for training/predictions
├── db.sqlite3                        # Local SQLite database
├── requirements.txt                  # Python dependencies
└── runtime.txt                       # Recommended Python runtime
```

---

## 🌐 Routes

### Remote User
- `/login/` — Remote user login
- `/Register1/` — Register a new account
- `/ViewYourProfile/` — User profile
- `/Predict_Theft_Status/` — Fraud prediction form
- `/Predict_Test_Inputs/` — Run predefined test input samples

### Service Provider
- `/serviceproviderlogin/` — Admin login
- `/train_model/` — Train and compare ML models
- `/Statistical/` — Model performance dashboard
- `/Download_Predicted_DataSets/` — Download predictions as Excel

---

## 🛠️ Built With

- **Python 3.11**
- **Django 3.2**
- **pandas**
- **scikit-learn**
- **numpy**
- **xlwt**
- **python-decouple**
- **gunicorn**

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ysathyasai/Composite_Behavioral_Modeling.git
cd Composite_Behavioral_Modeling
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Run migrations

```bash
cd composite_behavioral_modeling
python manage.py migrate
```

### 5. Start the server

```bash
python manage.py runserver
```

### 6. Access the app

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/login/`
- `http://127.0.0.1:8000/Register1/`
- `http://127.0.0.1:8000/serviceproviderlogin/`

---

## 🔐 Admin Login

- **Username:** `Admin`
- **Password:** `Admin`

---

## 📌 Notes

- `Datasets.csv` must be present in the repository root for predictions and training.
- `Predict_Test_Inputs.csv` provides a set of predefined sample values and a dedicated `/Predict_Test_Inputs/` page for quick testing.
- Model training is currently performed through the service provider views for demo purposes.
- The app uses simple session handling and should be migrated to Django authentication in production.

> [!NOTE]
> Hardware Resource Constraints: This project may return an Internal Server Error during model execution. As this is an academic project hosted on the Render Free Tier, the 0.5 CPU allocation and limited RAM are insufficient to process the model's computational requirements. Running the project locally is suggested to utilize the model effectively.

---

## 🚧 Future Improvements

- Add Django built-in authentication and password hashing
- Add automated tests for auth, prediction, and training flows
- Move training to a background worker or scheduled job

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✅ Why This Project Works

- Balanced remote user and admin workflows
- ML-driven fraud classification and reporting
- Export support for audit and analysis
- Easy local setup for reviewers
- Designed for submission-ready presentation

---

## 📬 Contact

For enhancements, you can extend the model, secure authentication, or add a more comprehensive dataset pipeline.

<p align="center">
  Made with ❤️ by <a href="https://github.com/ysathyasai">ysathyasai</a> <br> and CBM team
</p>
