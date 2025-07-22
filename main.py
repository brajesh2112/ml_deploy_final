import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import time

# Load trained logistic regression model
model = joblib.load("logreg_model.pkl")

# Load encoders used during preprocessing
label_encoders = joblib.load("label_encoders.pkl")
onehot_encoders = joblib.load("onehot_encoders.pkl")

# Categorical and numerical features
categorical_features = ['Gender', 'City', 'Profession', 'Degree', 'Sleep Duration',
                        'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
numerical_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
                      'Work/Study Hours', 'Financial Stress']

def preprocess_input(data):
    df = pd.DataFrame([data], columns=categorical_features + numerical_features)
    
    for col in categorical_features:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
        elif col in onehot_encoders:
            encoded_arr = onehot_encoders[col].transform(df[[col]])
            encoded_df = pd.DataFrame(encoded_arr, columns=onehot_encoders[col].get_feature_names_out([col]))
            df.drop(columns=[col], inplace=True)
            df = pd.concat([df, encoded_df], axis=1)
    
    return df

def predict_depression(input_data):
    processed_input = preprocess_input(input_data)
    missing_cols = set(model.feature_names_in_) - set(processed_input.columns)
    for col in missing_cols:
        processed_input[col] = 0  # Add missing columns as zeros
    processed_input = processed_input[model.feature_names_in_]
    
    #st.write("Preprocessed input preview:")
    #st.write(processed_input)
    
    prediction = model.predict(processed_input)[0]
    return "Depressed" if prediction == 1 else "Not Depressed"

# Streamlit UI
# Set page config
st.set_page_config(page_title="Student Depression Prediction", layout="wide")#wide/centered

# Main Header
st.image("https://www.shutterstock.com/image-vector/kids-stress-depression-concept-depressed-600nw-2071989683.jpg", use_container_width=True)
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B; font-size: 36px;'>
        🎓 Student Depression Prediction
    </h1>
    <p style='text-align: center; font-size: 20px;'>
        Enter your details below to get a mental health prediction using machine learning.
    </p>
""", unsafe_allow_html=True)



# Sidebar Image
with st.sidebar:
    st.image("https://mindvoyage.in/wp-content/uploads/2022/06/DepressionbyState.jpg",use_container_width=True )# use_container_width=True
    st.markdown("🧠 **Mental Health Awareness**")
    st.markdown("Depression is increasing globally. Let's break the stigma!")
    



#input
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 16, 55, 22)
city = st.selectbox("City", ['Agra', 'Ahmedabad', 'Bangalore', 'Bhopal', 'Chennai', 'Delhi', 'Faridabad',
                             'Ghaziabad', 'Hyderabad', 'Indore', 'Jaipur', 'Kalyan', 'Kanpur', 'Khaziabad',
                             'Kolkata', 'Lucknow', 'Ludhiana', 'Meerut', 'Mumbai', 'Nagpur', 'Nashik',
                             'Patna', 'Pune', 'Rajkot', 'Srinagar', 'Surat', 'Thane', 'Vadodara',
                             'Varanasi', 'Vasai-Virar', 'Visakhapatnam']
)

profession = st.selectbox("Profession", ['Student', 'Civil Engineer', 'Architect', 'UX/UI Designer', 
 'Digital Marketer', 'Content Writer', 'Educational Consultant', 'Teacher',
 'Manager', 'Chef', 'Doctor', 'Lawyer', 'Entrepreneur', 'Pharmacist']
)
academic_pressure = st.slider("Academic Pressure", 0, 5, 3)
cgpa = st.slider("CGPA", 0.0, 10.0, 6.5)
study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3)
sleep_duration = st.selectbox("Sleep Duration", ['5-6 hours','Less than 5 hours','7-8 hours','More than 8 hours','Others'])
dietary_habits = st.selectbox("Dietary Habits", ['Healthy','Moderate','Unhealthy','Others'])
degree = st.selectbox("Degree", ['B.Arch', 'B.Com', 'B.Ed', 'B.Pharm', 'B.Tech', 'BBA', 'BCA', 'BE', 'BHM', 'BSc', 'BA', 'Class 12', 'LLB', 'LLM', 'M.Com', 'M.Ed', 'M.Pharm', 'M.Tech', 'MBA', 'MBBS', 'MCA', 'MD', 'ME', 'MHM', 'MA', 'MSc', 'Others', 'PhD']
)
suicidal_thoughts = st.selectbox("Suicidal Thoughts", ['Yes', 'No'])
work_study_hours = st.slider("Work/Study Hours", 0, 12, 5)
financial_stress = st.slider("Financial Stress", 1, 5, 3)
family_history = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])

if st.button("Predict Depression"):
    input_data = [ gender, city, profession, degree, sleep_duration, dietary_habits,
        suicidal_thoughts, family_history,  # categorical
        age, academic_pressure, cgpa, study_satisfaction,
        work_study_hours, financial_stress   # numerical
    ]



    prediction = predict_depression(input_data)
    st.success(f"Prediction: {prediction}")
    if prediction == "Depressed":
        st.error("🧠 You might be going through a tough time.")
        st.markdown("💬 *“You are not alone. Please consider reaching out.”*")
        st.markdown("📞 **Helpline:** Call iCall – 9152987821 (available 24x7)")
        st.markdown("🔗 [Visit Mental Health India](https://mentalhealthindia.org)")
    else:
        st.balloons()
        st.markdown("😊 *“Keep going, you're doing great!”*")
        st.markdown("💡 Stay connected, talk to friends, and take regular breaks.")


st.write("This model uses machine learning to analyze factors contributing to student depression.")

# Real-time Clock
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"🕒 **Current Time:** {current_time}")

# Page view count (simple session-based)
if 'count' not in st.session_state:
    st.session_state.count = 1
else:
    st.session_state.count += 1

st.markdown(f"👀 **Page Views this session:** {st.session_state.count}")

# Footer
st.markdown("---")
st.markdown("💙 *Created with care to promote mental wellness among students.*")





