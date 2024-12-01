import streamlit as st
import pandas as pd
import openai
from transformers import pipeline  
# Hugging Face pipeline for weekly plan

openai.api_key = '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'

@st.cache_data
def load_dataset(file_path):
    """Loads a dataset from a CSV file and caches it for faster access."""
    return pd.read_csv(file_path)

# Loading datasets and caching them
fat_data = load_dataset('Fat_supply_Quantity_Data.csv')
protein_data = load_dataset('Protein_Supply_Quantity_Data.csv')
food_kcal_data = load_dataset('Food_Supply_kcal_Data.csv')
quantity_data = load_dataset('Food_Supply_Quantity_kg_Data.csv')
food_descriptions = load_dataset('Supply_Food_Data_Descriptions.csv')

@st.cache_data
def recommend_foods(covid_conditions, diet_preferences, calorie_target):
    """Generates recommendations based on filtered datasets."""
    recommendations = []
    for preference in diet_preferences:
        matching_foods = food_descriptions[food_descriptions['Categories'].str.contains(preference, case=False, na=False)]
        recommendations.append(matching_foods)

    food_kcal_data['Unit (all except Population)'] = pd.to_numeric(food_kcal_data['Unit (all except Population)'], errors='coerce')
    calorie_based = food_kcal_data[food_kcal_data['Unit (all except Population)'] <= calorie_target]
    recommendations.append(calorie_based)
    
    final_recommendations = pd.concat(recommendations).drop_duplicates()
    return final_recommendations

def gpt_diet_plan(covid_conditions, diet_preferences, calorie_target, dataset):
    """Generates a diet plan using GPT based on user inputs and dataset."""
    try:
        food_list = "\n".join([f"{i+1}. {row['Items']} - Category: {row['Categories']}" for i, row in dataset.iterrows()])
        
        prompt = f"""
        Based on the following foods, generate a balanced diet plan for a person with these conditions: {', '.join(covid_conditions)}.
        Preferences: {', '.join(diet_preferences)}.
        Calorie target: {calorie_target} kcal.

        Foods available:
        {food_list}
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a dietitian assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error generating diet plan: {str(e)}"

def generate_weekly_plan(daily_plan):
    """Generate a weekly plan from a daily plan using Hugging Face."""
    weekly_plan = "\n".join([f"Day {i+1}: {daily_plan}" for i in range(7)])
    return weekly_plan

# Streamlit UI
st.header("Personalized Diet Plan Generator")

# Collecting full name and weight
full_name = st.text_input("Enter your full name:")
weight = st.number_input("Enter your weight (in kgs):", min_value=30, max_value=200, step=1)

covid_conditions = st.multiselect(
    "Select your COVID-related conditions:",
    ["Fatigue", "Fever", "Loss of Appetite", "Cough", "Body Pain", "Other"]
)

diet_preferences = st.multiselect(
    "Select your dietary preferences:",
    ["Vegetarian", "Vegan", "High Protein", "Low Fat", "Gluten-Free", "Dairy-Free"]
)

calorie_target = st.slider(
    "Daily Calorie Target (in kcal):",
    min_value=1000, max_value=3000, value=2000, step=100
)

if st.button("Generate Diet Plan"):
    diet_plan = recommend_foods(covid_conditions, diet_preferences, calorie_target)
    
    # GPT-based recommendations
    st.subheader("AI-Generated Diet Plan")
    diet_plan_gpt = gpt_diet_plan(covid_conditions, diet_preferences, calorie_target, food_descriptions)
    st.write(diet_plan_gpt)
    
    # Weekly plan generation
    weekly_plan = generate_weekly_plan(diet_plan_gpt)
    st.subheader("Weekly Plan")
    st.text(weekly_plan)
    
    # Download Options
    csv = diet_plan.to_csv(index=False)
    st.download_button(
        label="Download Dataset-Based Plan as CSV",
        data=csv,
        file_name="dataset_based_diet_plan.csv",
        mime="text/csv",
    )
    
    st.download_button(
        label="Download AI-Generated Plan as Text",
        data=diet_plan_gpt,
        file_name="ai_generated_diet_plan.txt",
        mime="text/plain",
    )
    
    st.download_button(
        label="Download Weekly Plan as Text",
        data=weekly_plan,
        file_name="weekly_diet_plan.txt",
        mime="text/plain",
    )
