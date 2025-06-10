import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

# Load dan simpan ulang model
@st.cache
def load_model():
    model = joblib.load('model.pkl')
    return model

model = load_model()

# Load the cleaned data
@st.cache_data
def load_data():
    return pd.read_csv('data_cleaned.csv')

data_cleaned = load_data()

# Title and description
st.title("Student Dropout Prediction System")

# Function to get user input
def get_user_input():
    # Create expandable sections for better organization
    with st.expander("Personal Information", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.radio("Gender", ["Female", "Male"])
            age = st.slider("Age at Enrollment", 17, 70, 20)
            marital_status = st.selectbox("Marital Status", [
                "Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"
            ])
            nationality = st.selectbox("Nationality", [
                "Portuguese", "German", "Spanish", "Italian", "Dutch", "English", 
                "Lithuanian", "Angolan", "Cape Verdean", "Guinean", "Mozambican",
                "Santomean", "Turkish", "Brazilian", "Romanian", "Moldova (Republic of)",
                "Mexican", "Ukrainian", "Russian", "Cuban", "Colombian"
            ])
            
        with col2:
            displaced = st.radio("Displaced Student", ["No", "Yes"])
            educational_special_needs = st.radio("Educational Special Needs", ["No", "Yes"])
            debtor = st.radio("Debtor", ["No", "Yes"])
            tuition_fees_up_to_date = st.radio("Tuition Fees Up to Date", ["No", "Yes"])
            scholarship_holder = st.radio("Scholarship Holder", ["No", "Yes"])
            international = st.radio("International Student", ["No", "Yes"])

    with st.expander("Academic Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            application_mode = st.selectbox("Application Mode", [
                "1st phase - general contingent", "Ordinance No. 612/93", 
                "1st phase - special contingent (Azores Island)", 
                "Holders of other higher courses", "Ordinance No. 854-B/99", 
                "International student (bachelor)", 
                "1st phase - special contingent (Madeira Island)", 
                "2nd phase - general contingent", "3rd phase - general contingent", 
                "Ordinance No. 533-A/99, item b2) (Different Plan)", 
                "Ordinance No. 533-A/99, item b3 (Other Institution)", 
                "Over 23 years old", "Transfer", "Change of course", 
                "Technological specialization diploma holders", 
                "Change of institution/course", "Short cycle diploma holders", 
                "Change of institution/course (International)"
            ])
            application_order = st.slider("Application Order (0 = first choice)", 0, 9, 0)
            course = st.selectbox("Course", [
                "Biofuel Production Technologies", "Animation and Multimedia Design", 
                "Social Service (evening attendance)", "Agronomy", 
                "Communication Design", "Veterinary Nursing", 
                "Informatics Engineering", "Equinculture", "Management", 
                "Social Service", "Tourism", "Nursing", "Oral Hygiene", 
                "Advertising and Marketing Management", 
                "Journalism and Communication", "Basic Education", 
                "Management (evening attendance)"
            ])
            
        with col2:
            attendance = st.radio("Daytime/Evening Attendance", ["Daytime", "Evening"])
            previous_qualification = st.selectbox("Previous Qualification", [
                "Secondary education", "Higher education - bachelor's degree", 
                "Higher education - degree", "Higher education - master's", 
                "Higher education - doctorate", "Frequency of higher education", 
                "12th year of schooling - not completed", 
                "11th year of schooling - not completed", 
                "Other - 11th year of schooling", "10th year of schooling", 
                "10th year of schooling - not completed", 
                "Basic education 3rd cycle (9th/10th/11th year) or equiv.", 
                "Basic education 2nd cycle (6th/7th/8th year) or equiv.", 
                "Technological specialization course", 
                "Higher education - degree (1st cycle)", 
                "Professional higher technical course", 
                "Higher education - master (2nd cycle)"
            ])
            previous_qualification_grade = st.slider("Previous Qualification Grade", 0, 200, 120)
            admission_grade = st.slider("Admission Grade", 0, 200, 120)

    with st.expander("Family Background"):
        col1, col2 = st.columns(2)
        
        with col1:
            mothers_qualification = st.selectbox("Mother's Qualification", [
                "Secondary Education - 12th Year of Schooling or Eq.", 
                "Higher Education - Bachelor's Degree", 
                "Higher Education - Degree", "Higher Education - Master's", 
                "Higher Education - Doctorate", "Frequency of Higher Education", 
                "12th Year of Schooling - Not Completed", 
                "11th Year of Schooling - Not Completed", "7th Year (Old)", 
                "Other - 11th Year of Schooling", "10th Year of Schooling", 
                "General commerce course", 
                "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.", 
                "Technical-professional course", "7th year of schooling", 
                "2nd cycle of the general high school course", 
                "9th Year of Schooling - Not Completed", "8th year of schooling", 
                "Unknown", "Can't read or write", 
                "Can read without having a 4th year of schooling", 
                "Basic education 1st cycle (4th/5th year) or equiv.", 
                "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.", 
                "Technological specialization course", 
                "Higher education - degree (1st cycle)", 
                "Specialized higher studies course", 
                "Professional higher technical course", 
                "Higher Education - Master (2nd cycle)", 
                "Higher Education - Doctorate (3rd cycle)"
            ])
            fathers_qualification = st.selectbox("Father's Qualification", [
                "Secondary Education - 12th Year of Schooling or Eq.", 
                "Higher Education - Bachelor's Degree", 
                "Higher Education - Degree", "Higher Education - Master's", 
                "Higher Education - Doctorate", "Frequency of Higher Education", 
                "12th Year of Schooling - Not Completed", 
                "11th Year of Schooling - Not Completed", "7th Year (Old)", 
                "Other - 11th Year of Schooling", 
                "2nd year complementary high school course", 
                "10th Year of Schooling", "General commerce course", 
                "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.", 
                "Complementary High School Course", "Technical-professional course", 
                "Complementary High School Course - not concluded", 
                "7th year of schooling", 
                "2nd cycle of the general high school course", 
                "9th Year of Schooling - Not Completed", "8th year of schooling", 
                "General Course of Administration and Commerce", 
                "Supplementary Accounting and Administration", "Unknown", 
                "Can't read or write", 
                "Can read without having a 4th year of schooling", 
                "Basic education 1st cycle (4th/5th year) or equiv.", 
                "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.", 
                "Technological specialization course", 
                "Higher education - degree (1st cycle)", 
                "Specialized higher studies course", 
                "Professional higher technical course", 
                "Higher Education - Master (2nd cycle)", 
                "Higher Education - Doctorate (3rd cycle)"
            ])
            
        with col2:
            mothers_occupation = st.selectbox("Mother's Occupation", [
                "Student", 
                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
                "Specialists in Intellectual and Scientific Activities", 
                "Intermediate Level Technicians and Professions", 
                "Administrative staff", 
                "Personal Services, Security and Safety Workers and Sellers", 
                "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry", 
                "Skilled Workers in Industry, Construction and Craftsmen", 
                "Installation and Machine Operators and Assembly Workers", 
                "Unskilled Workers", "Armed Forces Professions", 
                "Other Situation", "(blank)", "Health professionals", 
                "Teachers", 
                "Specialists in information and communication technologies (ICT)", 
                "Intermediate level science and engineering technicians and professions", 
                "Technicians and professionals, of intermediate level of health", 
                "Intermediate level technicians from legal, social, sports, cultural and similar services", 
                "Office workers, secretaries in general and data processing operators", 
                "Data, accounting, statistical, financial services and registry-related operators", 
                "Other administrative support staff", "Personal service workers", 
                "Sellers", "Personal care workers and the like", 
                "Skilled construction workers and the like, except electricians", 
                "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like", 
                "Workers in food processing, woodworking, clothing and other industries and crafts", 
                "Cleaning workers", 
                "Unskilled workers in agriculture, animal production, fisheries and forestry", 
                "Unskilled workers in extractive industry, construction, manufacturing and transport", 
                "Meal preparation assistants"
            ])
            fathers_occupation = st.selectbox("Father's Occupation", [
                "Student", 
                "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
                "Specialists in Intellectual and Scientific Activities", 
                "Intermediate Level Technicians and Professions", 
                "Administrative staff", 
                "Personal Services, Security and Safety Workers and Sellers", 
                "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry", 
                "Skilled Workers in Industry, Construction and Craftsmen", 
                "Installation and Machine Operators and Assembly Workers", 
                "Unskilled Workers", "Armed Forces Professions", 
                "Other Situation", "(blank)", "Armed Forces Officers", 
                "Armed Forces Sergeants", "Other Armed Forces personnel", 
                "Directors of administrative and commercial services", 
                "Hotel, catering, trade and other services directors", 
                "Specialists in the physical sciences, mathematics, engineering and related techniques", 
                "Health professionals", "Teachers", 
                "Specialists in finance, accounting, administrative organization, public and commercial relations", 
                "Intermediate level science and engineering technicians and professions", 
                "Technicians and professionals, of intermediate level of health", 
                "Intermediate level technicians from legal, social, sports, cultural and similar services", 
                "Information and communication technology technicians", 
                "Office workers, secretaries in general and data processing operators", 
                "Data, accounting, statistical, financial services and registry-related operators", 
                "Other administrative support staff", "Personal service workers", 
                "Sellers", "Personal care workers and the like", 
                "Protection and security services personnel", 
                "Market-oriented farmers and skilled agricultural and animal production workers", 
                "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence", 
                "Skilled construction workers and the like, except electricians", 
                "Skilled workers in metallurgy, metalworking and similar", 
                "Skilled workers in electricity and electronics", 
                "Workers in food processing, woodworking, clothing and other industries and crafts", 
                "Fixed plant and machine operators", "Assembly workers", 
                "Vehicle drivers and mobile equipment operators", 
                "Unskilled workers in agriculture, animal production, fisheries and forestry", 
                "Unskilled workers in extractive industry, construction, manufacturing and transport", 
                "Meal preparation assistants", 
                "Street vendors (except food) and street service providers"
            ])

    with st.expander("Enrollment Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            curricular_units_1st_sem_credited = st.slider(
                "Curricular Units 1st Sem (credited)", 0, 20, 0)
            curricular_units_1st_sem_enrolled = st.slider(
                "Curricular Units 1st Sem (enrolled)", 0, 20, 5)
                
        with col2:
            curricular_units_1st_sem_evaluations = st.slider(
                "Curricular Units 1st Sem (evaluations)", 0, 20, 5)
            curricular_units_1st_sem_approved = st.slider(
                "Curricular Units 1st Sem (approved)", 0, 20, 5)
    
    # Create a dictionary with all the input data
    input_data = {
        'Gender': 1 if gender == "Male" else 0,
        'Age_at_enrollment': age,
        'Marital_status': ["Single", "Married", "Widower", "Divorced", "Facto Union", "Legally Separated"].index(marital_status) + 1,
        'Application_mode': [
            "1st phase - general contingent", "Ordinance No. 612/93", 
            "1st phase - special contingent (Azores Island)", 
            "Holders of other higher courses", "Ordinance No. 854-B/99", 
            "International student (bachelor)", 
            "1st phase - special contingent (Madeira Island)", 
            "2nd phase - general contingent", "3rd phase - general contingent", 
            "Ordinance No. 533-A/99, item b2) (Different Plan)", 
            "Ordinance No. 533-A/99, item b3 (Other Institution)", 
            "Over 23 years old", "Transfer", "Change of course", 
            "Technological specialization diploma holders", 
            "Change of institution/course", "Short cycle diploma holders", 
            "Change of institution/course (International)"
        ].index(application_mode) + 1,
        'Application_order': application_order,
        'Course': [
            "Biofuel Production Technologies", "Animation and Multimedia Design", 
            "Social Service (evening attendance)", "Agronomy", 
            "Communication Design", "Veterinary Nursing", 
            "Informatics Engineering", "Equinculture", "Management", 
            "Social Service", "Tourism", "Nursing", "Oral Hygiene", 
            "Advertising and Marketing Management", 
            "Journalism and Communication", "Basic Education", 
            "Management (evening attendance)"
        ].index(course) + 1,
        'Daytime/evening attendance': 1 if attendance == "Daytime" else 0,
        'Previous qualification': [
            "Secondary education", "Higher education - bachelor's degree", 
            "Higher education - degree", "Higher education - master's", 
            "Higher education - doctorate", "Frequency of higher education", 
            "12th year of schooling - not completed", 
            "11th year of schooling - not completed", 
            "Other - 11th year of schooling", "10th year of schooling", 
            "10th year of schooling - not completed", 
            "Basic education 3rd cycle (9th/10th/11th year) or equiv.", 
            "Basic education 2nd cycle (6th/7th/8th year) or equiv.", 
            "Technological specialization course", 
            "Higher education - degree (1st cycle)", 
            "Professional higher technical course", 
            "Higher education - master (2nd cycle)"
        ].index(previous_qualification) + 1,
        'Previous qualification (grade)': previous_qualification_grade,
        'Nacionality': [
            "Portuguese", "German", "Spanish", "Italian", "Dutch", "English", 
            "Lithuanian", "Angolan", "Cape Verdean", "Guinean", "Mozambican",
            "Santomean", "Turkish", "Brazilian", "Romanian", "Moldova (Republic of)",
            "Mexican", "Ukrainian", "Russian", "Cuban", "Colombian"
        ].index(nationality) + 1,
        'Mother\'s qualification': [
            "Secondary Education - 12th Year of Schooling or Eq.", 
            "Higher Education - Bachelor's Degree", 
            "Higher Education - Degree", "Higher Education - Master's", 
            "Higher Education - Doctorate", "Frequency of Higher Education", 
            "12th Year of Schooling - Not Completed", 
            "11th Year of Schooling - Not Completed", "7th Year (Old)", 
            "Other - 11th Year of Schooling", "10th Year of Schooling", 
            "General commerce course", 
            "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.", 
            "Technical-professional course", "7th year of schooling", 
            "2nd cycle of the general high school course", 
            "9th Year of Schooling - Not Completed", "8th year of schooling", 
            "Unknown", "Can't read or write", 
            "Can read without having a 4th year of schooling", 
            "Basic education 1st cycle (4th/5th year) or equiv.", 
            "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.", 
            "Technological specialization course", 
            "Higher education - degree (1st cycle)", 
            "Specialized higher studies course", 
            "Professional higher technical course", 
            "Higher Education - Master (2nd cycle)", 
            "Higher Education - Doctorate (3rd cycle)"
        ].index(mothers_qualification) + 1,
        'Father\'s qualification': [
            "Secondary Education - 12th Year of Schooling or Eq.", 
            "Higher Education - Bachelor's Degree", 
            "Higher Education - Degree", "Higher Education - Master's", 
            "Higher Education - Doctorate", "Frequency of Higher Education", 
            "12th Year of Schooling - Not Completed", 
            "11th Year of Schooling - Not Completed", "7th Year (Old)", 
            "Other - 11th Year of Schooling", 
            "2nd year complementary high school course", 
            "10th Year of Schooling", "General commerce course", 
            "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.", 
            "Complementary High School Course", "Technical-professional course", 
            "Complementary High School Course - not concluded", 
            "7th year of schooling", 
            "2nd cycle of the general high school course", 
            "9th Year of Schooling - Not Completed", "8th year of schooling", 
            "General Course of Administration and Commerce", 
            "Supplementary Accounting and Administration", "Unknown", 
            "Can't read or write", 
            "Can read without having a 4th year of schooling", 
            "Basic education 1st cycle (4th/5th year) or equiv.", 
            "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.", 
            "Technological specialization course", 
            "Higher education - degree (1st cycle)", 
            "Specialized higher studies course", 
            "Professional higher technical course", 
            "Higher Education - Master (2nd cycle)", 
            "Higher Education - Doctorate (3rd cycle)"
        ].index(fathers_qualification) + 1,
        'Mother\'s occupation': [
            "Student", 
            "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
            "Specialists in Intellectual and Scientific Activities", 
            "Intermediate Level Technicians and Professions", 
            "Administrative staff", 
            "Personal Services, Security and Safety Workers and Sellers", 
            "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry", 
            "Skilled Workers in Industry, Construction and Craftsmen", 
            "Installation and Machine Operators and Assembly Workers", 
            "Unskilled Workers", "Armed Forces Professions", 
            "Other Situation", "(blank)", "Health professionals", 
            "Teachers", 
            "Specialists in information and communication technologies (ICT)", 
            "Intermediate level science and engineering technicians and professions", 
            "Technicians and professionals, of intermediate level of health", 
            "Intermediate level technicians from legal, social, sports, cultural and similar services", 
            "Office workers, secretaries in general and data processing operators", 
            "Data, accounting, statistical, financial services and registry-related operators", 
            "Other administrative support staff", "Personal service workers", 
            "Sellers", "Personal care workers and the like", 
            "Skilled construction workers and the like, except electricians", 
            "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like", 
            "Workers in food processing, woodworking, clothing and other industries and crafts", 
            "Cleaning workers", 
            "Unskilled workers in agriculture, animal production, fisheries and forestry", 
            "Unskilled workers in extractive industry, construction, manufacturing and transport", 
            "Meal preparation assistants"
        ].index(mothers_occupation),
        'Father\'s occupation': [
            "Student", 
            "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers", 
            "Specialists in Intellectual and Scientific Activities", 
            "Intermediate Level Technicians and Professions", 
            "Administrative staff", 
            "Personal Services, Security and Safety Workers and Sellers", 
            "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry", 
            "Skilled Workers in Industry, Construction and Craftsmen", 
            "Installation and Machine Operators and Assembly Workers", 
            "Unskilled Workers", "Armed Forces Professions", 
            "Other Situation", "(blank)", "Armed Forces Officers", 
            "Armed Forces Sergeants", "Other Armed Forces personnel", 
            "Directors of administrative and commercial services", 
            "Hotel, catering, trade and other services directors", 
            "Specialists in the physical sciences, mathematics, engineering and related techniques", 
            "Health professionals", "Teachers", 
            "Specialists in finance, accounting, administrative organization, public and commercial relations", 
            "Intermediate level science and engineering technicians and professions", 
            "Technicians and professionals, of intermediate level of health", 
            "Intermediate level technicians from legal, social, sports, cultural and similar services", 
            "Information and communication technology technicians", 
            "Office workers, secretaries in general and data processing operators", 
            "Data, accounting, statistical, financial services and registry-related operators", 
            "Other administrative support staff", "Personal service workers", 
            "Sellers", "Personal care workers and the like", 
            "Protection and security services personnel", 
            "Market-oriented farmers and skilled agricultural and animal production workers", 
            "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence", 
            "Skilled construction workers and the like, except electricians", 
            "Skilled workers in metallurgy, metalworking and similar", 
            "Skilled workers in electricity and electronics", 
            "Workers in food processing, woodworking, clothing and other industries and crafts", 
            "Fixed plant and machine operators", "Assembly workers", 
            "Vehicle drivers and mobile equipment operators", 
            "Unskilled workers in agriculture, animal production, fisheries and forestry", 
            "Unskilled workers in extractive industry, construction, manufacturing and transport", 
            "Meal preparation assistants", 
            "Street vendors (except food) and street service providers"
        ].index(fathers_occupation),
        'Admission grade': admission_grade,
        'Displaced': 1 if displaced == "Yes" else 0,
        'Educational special needs': 1 if educational_special_needs == "Yes" else 0,
        'Debtor': 1 if debtor == "Yes" else 0,
        'Tuition fees up to date': 1 if tuition_fees_up_to_date == "Yes" else 0,
        'Scholarship holder': 1 if scholarship_holder == "Yes" else 0,
        'International': 1 if international == "Yes" else 0,
        'Curricular units 1st sem (credited)': curricular_units_1st_sem_credited,
        'Curricular units 1st sem (enrolled)': curricular_units_1st_sem_enrolled,
        'Curricular units 1st sem (evaluations)': curricular_units_1st_sem_evaluations,
        'Curricular units 1st sem (approved)': curricular_units_1st_sem_approved
    }
    
    # Convert the dictionary to DataFrame
    features = pd.DataFrame(input_data, index=[0])
    
    return features

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("Student Information Summary")
st.write(user_input)

# Make prediction
if st.button('Predict Dropout Risk'):
    try:
        # Get all columns from the cleaned data (excluding target)
        model_columns = [col for col in data_cleaned.columns if 'Status' not in col]
        
        # Create a DataFrame with all zeros and then fill in the user input
        prediction_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Fill in the values from user input
        for col in user_input.columns:
            if col in prediction_data.columns:
                prediction_data[col] = user_input[col]
        
        # Make prediction
        prediction = model.predict(prediction_data)
        prediction_proba = model.predict_proba(prediction_data)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error("High Risk of Dropout")
            st.write(f"Probability: {prediction_proba[0][1]*100:.2f}%")
            st.write("""
            **Recommendations:**
            - Consider academic counseling
            - Evaluate financial support options
            - Monitor academic progress closely
            - Provide additional learning resources
            """)
        else:
            st.success("Low Risk of Dropout")
            st.write(f"Probability: {prediction_proba[0][0]*100:.2f}%")
            st.write("""
            **Recommendations:**
            - Continue current academic support
            - Monitor for any changes in performance
            - Encourage participation in extracurricular activities
            """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")