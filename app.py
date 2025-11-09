import streamlit as st
import pandas as pd
import joblib
from db_helper import (
    init_db, register_user, verify_user, insert_prediction,
    fetch_history, insert_feedback
)
import matplotlib.pyplot as plt
# -------------------------------
# Initialize database & load model
# -------------------------------
init_db()

model_data = joblib.load("best_model.pkl")
if isinstance(model_data, tuple):
    model = model_data[0]
    encoder = model_data[1] if len(model_data) > 1 else None
else:
    model = model_data
    encoder = None

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="EDU2JOB", page_icon="🎓", layout="wide")

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 45px;
        color: #1A5276;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #34495E;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .about-section {
        background-color: #EBF5FB;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
        margin: auto;
        width: 80%;
    }
    .about-section h3 {
        color: #2C3E50;
    }
    .about-section p {
        color: #2C3E50;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 0;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #2E86C1;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR: AUTHENTICATION
# -------------------------------
st.sidebar.title("🔐 User Authentication")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

auth_choice = st.sidebar.radio("Choose Option", ["Login", "Sign Up"])

if st.session_state.user_id is None:
    if auth_choice == "Sign Up":
        username = st.sidebar.text_input("Username")
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register"):
            if register_user(username, email, password):
                st.sidebar.success("Registered successfully! Please log in.")
            else:
                st.sidebar.error("Username already exists!")

    elif auth_choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            user_id = verify_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.sidebar.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")

else:
    st.sidebar.success("Logged in Successfully!")
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.rerun()

# -------------------------------
# MAIN PAGE CONTENT
# -------------------------------
if st.session_state.user_id is None:
   # -------- BEFORE LOGIN (HOME PAGE) --------
   st.markdown("<h1 class='main-title'>🎓 EDU2JOB</h1>", unsafe_allow_html=True)
   st.markdown("<p class='subtitle'> Career Guidance for Students & Graduates</p>", unsafe_allow_html=True)

   st.markdown("""
    <div class='about-section'>
        <h3>About the App</h3>
        <p>
        <b>EDU2JOB</b> is your personal career companion — designed to bridge the gap between
        <b>education and employment</b>. Using advanced machine learning models, it analyzes
        your <b>academic qualifications</b>, <b>skills</b>, <b>certifications</b>, and <b>projects</b> to predict
        the most suitable <b>job role</b> for you.
        </p>
        <br>
        <h3>⚙️ How It Works</h3>
        <p style='text-align: left;'>
        🔹 <b>Step 1:</b> Sign up and log in to your EDU2JOB account.<br>
        🔹 <b>Step 2:</b> Enter your qualifications, skills, certifications, and experience.<br>
        🔹 <b>Step 3:</b> Our AI model processes your inputs and predicts your best-fit job role.<br>
        🔹 <b>Step 4:</b> View and track all your predictions in your personal dashboard.<br>
        🔹 <b>Step 5:</b> Rate your experience and help us improve through feedback.
        </p>
        <br>
        <h3> Why Choose EDU2JOB?</h3>
        <p style='text-align: left;'>
        ✅ Personalized Experience for Every User<br>
        ✅ Continuous Model Improvement with Feedback<br>
        ✅ Designed for Students, Graduates, and Job Seekers<br>
        ✅ Secure Login System with History Tracking
        </p>
        <br>
        <h3>💼 Transform Your Future!</h3>
        <p>
        Whether you're a <b>student exploring your career options</b> or a <b>graduate preparing
        for your first job</b>, EDU2JOB helps you make informed choices — faster and smarter.
        </p>
        <br>
        <p style='font-size:17px; font-weight:600; color:#1B4F72;'>
        👉 Ready to find your ideal career path?<br>
        <b>Login or Sign Up from the sidebar</b> to get your first prediction now! 
        </p>
    </div>
""", unsafe_allow_html=True)


else:
    # -------- AFTER LOGIN (MAIN APP) --------
    st.markdown("<h1 class='main-title'>🎓 EDU2JOB – Predict your jobrole from Educational background</h1>", unsafe_allow_html=True)
    st.divider()
    st.subheader("Enter Your Details")
    # -------------------------------
# Default input values
# -------------------------------
    if 'degree' not in st.session_state:
     st.session_state.degree = "Select Option"
    if 'major' not in st.session_state:
     st.session_state.major = "Select Option"
    if 'skill1' not in st.session_state:
     st.session_state.skill1 = "Select Option"
    if 'skill2' not in st.session_state:
     st.session_state.skill2 = "Select Option"
    if 'certification' not in st.session_state:
     st.session_state.certification = "Select Option"
    if 'experience_years' not in st.session_state:
     st.session_state.experience_years = 0
    if 'project_count' not in st.session_state:
     st.session_state.project_count = 0
    if 'internship' not in st.session_state:
     st.session_state.internship = "Select Option"
    if 'experience_level' not in st.session_state:
     st.session_state.experience_level = "Select Option"

    col1, col2 = st.columns(2)
    with col1:
        degree = st.selectbox("🎓 Degree", ["Select Option", "B.Tech", "M.Tech", "BCA", "MCA"], index=["Select Option", "B.Tech", "M.Tech", "BCA", "MCA"].index(st.session_state.degree))
        major = st.selectbox("📘 Major", ["Select Option", "AI", "CS", "IT", "ECE"], index=["Select Option", "AI", "CS", "IT", "ECE"].index(st.session_state.major))
        skill1 = st.selectbox("💡 Primary Skill", ["Select Option", "Python", "Java", "C++"], index=["Select Option", "Python", "Java", "C++"].index(st.session_state.skill1))
        skill2 = st.selectbox("🔧 Secondary Skill", ["Select Option", "SQL", "HTML", "React"], index=["Select Option", "SQL", "HTML", "React"].index(st.session_state.skill2))
        certification = st.selectbox("📜 Certification", ["Select Option", "AI Specialist", "Data Analyst", "Web Developer"], index=["Select Option", "AI Specialist", "Data Analyst", "Web Developer"].index(st.session_state.certification))

    with col2:
        experience_years = st.number_input("⌛ Experience (Years)", 0, 50, st.session_state.experience_years)
        project_count = st.number_input("📁 Project Count", 0, 50, st.session_state.project_count)
        internship = st.selectbox("🎯 Internship", ["Select Option", "Yes", "No"], index=["Select Option", "Yes", "No"].index(st.session_state.internship))
        experience_level = st.selectbox("⭐ Experience Level", ["Select Option", "Beginner", "Intermediate", "Expert"], index=["Select Option", "Beginner", "Intermediate", "Expert"].index(st.session_state.experience_level))

    st.markdown("---")
    # -------------------------------
    # Clear / Reset Buttons
    # -------------------------------
    col1, col2 = st.columns(2)

    with col2:
     if st.button("Reset Input Form"):
        st.session_state.degree = "Select Option"
        st.session_state.major = "Select Option"
        st.session_state.skill1 = "Select Option"
        st.session_state.skill2 = "Select Option"
        st.session_state.certification = "Select Option"
        st.session_state.experience_years = 0
        st.session_state.project_count = 0
        st.session_state.internship = "Select Option"
        st.session_state.experience_level = "Select Option"
        st.rerun()

# -------------------------------
# Prepare input data for prediction
# -------------------------------
    input_data = pd.DataFrame({
     "Degree": [degree],
     "Major": [major],
     "Skill1": [skill1],
     "Skill2": [skill2],
     "Certification": [certification],
     "ExperienceYears": [experience_years],
     "ProjectCount": [project_count],
     "Internship": [internship],
     "ExperienceLevel": [experience_level]
   })

    with col1:
     if st.button("🔍 Predict Job Role"):
        
    # Predict probabilities
      if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0] 
        class_indices = probs.argsort()[::-1] 
        top_n = 3
        top_classes = class_indices[:top_n]
        
        results = []
        for idx in top_classes:
            class_name = encoder.inverse_transform([idx])[0] if encoder else model.classes_[idx]
            confidence = probs[idx] * 100
            results.append((class_name, confidence))
        
       
        st.success("Top Job Role Matches:")
        for i, (role, conf) in enumerate(results, 1):
            st.write(f"{i}. {role} — {conf:.2f}% confidence")

       
        row = input_data.iloc[0].to_dict()
        row["predicted_label"] = results[0][0]  
        insert_prediction(st.session_state.user_id, row)
        st.info("Saved your top prediction to history.")
        with col2:
         roles = [r[0] for r in results]
         confidences = [r[1] for r in results]

         fig, ax = plt.subplots()
         ax.barh(roles, confidences, color='#2E86C1')
         ax.set_xlabel("Confidence (%)")
         ax.set_title("Top Job Role Matches")
         ax.invert_yaxis()  # highest at top
         st.pyplot(fig)
      else:
        # Fallback: if model does not support predict_proba
        prediction = model.predict(input_data)
        predicted_role = encoder.inverse_transform(prediction)[0] if encoder else prediction[0]
        st.success(f"Predicted Job Role: **{predicted_role}**")
        
        row = input_data.iloc[0].to_dict()
        row["predicted_label"] = predicted_role
        insert_prediction(st.session_state.user_id, row)
        st.info("Saved to your history.")

    
    st.divider()

    st.subheader("📚 Your Prediction History")
    rows = fetch_history(st.session_state.user_id)
    if rows:
        df_hist = pd.DataFrame(rows, columns=[
            "id","user_id","timestamp","degree","major","skill1","skill2","certification",
            "experience_years","project_count","internship","experience_level","predicted_label"
        ])
        st.dataframe(df_hist, use_container_width=True)
       
        if st.button("🗑️ Clear Prediction History"):
          from db_helper import clear_history 
          clear_history(st.session_state.user_id)
          st.success("✅ Your prediction history has been cleared!")
          st.rerun()
    else:
        st.info("No prediction history yet.")

    st.divider()

    

    st.subheader("⭐ Feedback & Rating")
    rating = st.slider("Rate this app", 1, 5, 4)
    comment = st.text_area(" Your Feedback")
    if st.button("Submit Feedback "):
        insert_feedback(st.session_state.user_id, rating, comment)
        st.success(" Thank you for your valuable feedback!")
