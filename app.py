import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset using st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv('dataset1.csv')

# Train the model using st.cache_data
@st.cache_data
def train_model(df):
    X = df.drop(columns=['Output'])
    y = df['Output']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Load the dataset
df = load_data()

# Train the model
model, X_test, y_test = train_model(df)

# Evaluate model accuracy
@st.cache_data
def evaluate_model(_model, X_test, y_test):
    y_pred = _model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

accuracy = evaluate_model(model, X_test, y_test)

# Streamlit app
st.title('Crop Output Prediction')
st.write('Accuracy of the model:', accuracy*100,"%")

st.sidebar.header('Input Parameters')

# Sidebar inputs
N = st.sidebar.number_input('Nitrogen amount', min_value=df['N'].min(), max_value=df['N'].max())
P = st.sidebar.number_input('Phosphorus content', min_value=df['P'].min(), max_value=df['P'].max())
K = st.sidebar.number_input('Potassium content', min_value=df['K'].min(), max_value=df['K'].max())
pH = st.sidebar.number_input('pH', min_value=df['pH'].min(), max_value=df['pH'].max())
EC = st.sidebar.number_input('Electrical conductivity', min_value=df['EC'].min(), max_value=df['EC'].max())
OC = st.sidebar.number_input('Oxygen content', min_value=df['OC'].min(), max_value=df['OC'].max())
S = st.sidebar.number_input('Sulphur content', min_value=df['S'].min(), max_value=df['S'].max())
Zn = st.sidebar.number_input('Zinc content', min_value=df['Zn'].min(), max_value=df['Zn'].max())
Fe = st.sidebar.number_input('Iron content', min_value=df['Fe'].min(), max_value=df['Fe'].max())
Cu = st.sidebar.number_input('Copper content', min_value=df['Cu'].min(), max_value=df['Cu'].max())
Mn = st.sidebar.number_input('Manganese content', min_value=df['Mn'].min(), max_value=df['Mn'].max())

# Predict output
input_data = pd.DataFrame({
    'N': [N], 'P': [P], 'K': [K], 'pH': [pH], 'EC': [EC],
    'OC': [OC], 'S': [S], 'Zn': [Zn], 'Fe': [Fe], 'Cu': [Cu], 'Mn': [Mn]
})

# Ensure columns match those used during training
expected_columns = df.drop(columns=['Output']).columns
input_data = input_data.reindex(columns=expected_columns, fill_value=0)  # Fill with 0 for missing columns

prediction = model.predict(input_data)[0]

output_mapping = {0: 'Not Favorable', 1: 'Normal', 2: 'Highly Favorable'}
prediction_label = output_mapping[prediction]

st.subheader('Prediction:')
st.write('The predicted output category is:', prediction_label)
