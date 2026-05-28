# 🥗 Personalized Diet Plan Generator

A Streamlit web application that generates personalized diet plans based on COVID-related health conditions, dietary preferences, and calorie targets — powered by GPT-4 and global food supply data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Notes & Limitations](#notes--limitations)

---

## Overview

This app combines real-world food supply data (sourced from the FAO) with OpenAI's GPT-4 to recommend personalized, condition-aware diet plans. Users can input their health conditions, dietary preferences, and daily calorie goals to receive a tailored daily and weekly meal plan.

---

## Features

- **Condition-aware recommendations** — filters food suggestions based on COVID-related symptoms (fatigue, fever, loss of appetite, etc.)
- **Dietary preference filtering** — supports vegetarian, vegan, high-protein, low-fat, gluten-free, and dairy-free options
- **Calorie targeting** — plans are generated within a user-defined daily calorie range (1,000–3,000 kcal)
- **AI-generated plans** — GPT-4 produces a readable, structured daily diet plan
- **Weekly plan expansion** — daily plans are automatically extended into a full 7-day schedule
- **Downloadable outputs** — export plans as CSV or plain text files

---

## Datasets

The app uses five CSV files derived from FAO global food supply data:

| File | Description |
|---|---|
| `Fat_Supply_Quantity_Data.csv` | Fat supply per food category by country |
| `Protein_Supply_Quantity_Data.csv` | Protein supply per food category by country |
| `Food_Supply_kcal_Data.csv` | Caloric supply per food category by country |
| `Food_Supply_Quantity_kg_Data.csv` | Food quantity (kg) per category by country |
| `Supply_Food_Data_Descriptions.csv` | Category-to-item mappings for food groups |

Each dataset also includes COVID-related statistics (confirmed cases, deaths, recovered, active) and obesity/undernourishment rates per country.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/diet-plan-generator.git
cd diet-plan-generator
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set your OpenAI API key**

Open `app.py` and replace the placeholder with your actual API key:
```python
openai.api_key = 'your-api-key-here'
```

> ⚠️ For production use, store your key in an environment variable or a `.env` file instead of hardcoding it.

**4. Run the app**
```bash
streamlit run app.py
```

---

## Usage

1. Enter your **full name** and **weight**
2. Select any **COVID-related conditions** you are experiencing
3. Choose your **dietary preferences**
4. Set your **daily calorie target** using the slider
5. Click **"Generate Diet Plan"** to receive:
   - A dataset-filtered food recommendation list
   - An AI-written daily diet plan
   - A 7-day weekly plan
6. Download any of the outputs using the provided buttons

---

## Project Structure

```
diet-plan-generator/
│
├── app.py                            # Main Streamlit application
├── requirements.txt                  # Python dependencies
│
├── Fat_Supply_Quantity_Data.csv
├── Protein_Supply_Quantity_Data.csv
├── Food_Supply_kcal_Data.csv
├── Food_Supply_Quantity_kg_Data.csv
└── Supply_Food_Data_Descriptions.csv
```

---

## Dependencies

```
streamlit
pandas
scikit-learn
openai==0.28
matplotlib
transformers
```

Install all at once with:
```bash
pip install -r requirements.txt
```

---

## Notes & Limitations

- The app uses **OpenAI API v0.28** — if upgrading to a newer version, the `ChatCompletion.create()` call syntax will need to be updated.
- The **API key is currently hardcoded** in `app.py`. This is a security risk; use environment variables before deploying publicly.
- The weekly plan generator repeats the daily plan across 7 days — future versions could vary meals per day.
- Dataset coverage is **country-level and aggregate**; individual food item nutritional detail is limited to the FAO categories listed in `Supply_Food_Data_Descriptions.csv`.

---

## License

This project is for educational and research purposes. Please ensure compliance with OpenAI's usage policies and the FAO's data terms when deploying or distributing.
