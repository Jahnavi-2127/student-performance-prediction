import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data and define X (input) and y (output)
df = pd.read_csv('student_data.csv')
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = df['Performance_Index']

# Split 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained 'brain' to a file
joblib.dump(model, 'student_model.pkl')
print("✅ AI Model Trained and Saved as student_model.pkl!")