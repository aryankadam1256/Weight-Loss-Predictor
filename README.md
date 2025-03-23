# Weight-Loss-Predictor
🏅 Weight Loss Prediction Model

📊 Overview

This project predicts weight loss based on user inputs like BMI, Age, Gender, Exercise Duration, Number of Days, and Exercise Type. It is built as an interactive web application using HTML, CSS, and JavaScript, with predictions powered by a pre-trained Ridge Regression model in Python.

💡 Features

✅ Predicts weight loss (kg) based on input parameters

✅ Uses Machine Learning (Ridge Regression) for accuracy

✅ Simple & responsive web interface

✅ Styled using Poppins & Baloo Bhai 2 fonts

🗂 Project Structure

📝 Weight_Loss_Prediction
│── 📚 index.html          # Webpage structure (UI)
│── 📚 styles.css          # CSS for styling the webpage
│── 📚 script.js           # JavaScript for calculations
│── 📚 Weight_Loss_Calculation.py  # Machine Learning model training
│── 📚 README.md           # Documentation

📲 Web Application (Frontend)

The web interface is built using HTML, CSS, and JavaScript.

📄 index.html (Frontend UI)

This file contains the structure of the webpage, including:

Form Inputs → BMI, Age, Gender, Exercise Duration, etc.

Prediction Display → Shows the predicted weight loss.

Button → Calls JavaScript function to calculate weight loss.

👉 Key Code Snippet (Form Section):

<form id="inputForm">
    <label for="bmi">BMI:</label>
    <input type="number" id="bmi" name="bmi" required>

    <label for="age">Age:</label>
    <input type="number" id="age" name="age" required>

    <button type="button" onclick="predictWeightLoss()">Predict Weight Loss</button>
</form>

🎨 styles.css (Styling & UI)

This file contains the styling for the webpage, making it visually appealing and responsive.

👉 Key Features:

Modern Blue Theme (#3498db)

Smooth Button Hover Effects

Responsive Design for Different Screens

👉 Key Code Snippet (Button Styling):

button {
    background-color: #3498db;
    color: white;
    font-weight: 600;
    padding: 12px 25px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    cursor: pointer;
    transition: background 0.3s, transform 0.2s;
}
button:hover {
    background-color: #2980b9;
    transform: scale(1.05);
}

🧐 Backend (JavaScript & Machine Learning)

The prediction logic is handled using JavaScript, which applies a trained ML model.

📝 script.js (Prediction Logic)

This file takes user input, normalizes it, and applies pre-trained model coefficients to predict weight loss.

👉 Key Features:

Uses Machine Learning Coefficients

Normalizes Input Data

Caps Predictions Between 0-15 kg

👉 Key Code Snippet (Prediction Calculation):

const coefficients = [0.0575, -0.2944, -0.0133, 1.4067, 0.9443, 0.0021];
const intercept = 5.28819;

function predictWeightLoss() {
    const bmi = parseFloat(document.getElementById("bmi").value);
    const age = parseInt(document.getElementById("age").value);

    let predictedWeightLoss = intercept;
    for (let i = 0; i < coefficients.length; i++) {
        predictedWeightLoss += coefficients[i] * scaledInput[i];
    }

    predictedWeightLoss = Math.max(0, Math.min(15, predictedWeightLoss));
    document.getElementById("predictionOutput").innerText = predictedWeightLoss.toFixed(2);
}

🤖 Weight_Loss_Calculation.py (Machine Learning Model)

This Python file trains the model using Ridge Regression on a dataset with BMI, Age, Gender, Exercise Duration, etc.

👉 Key Features:

Uses scikit-learn Ridge Regression

Standardizes data using StandardScaler

Generates learned coefficients for JavaScript to use

👉 Key Code Snippet (Training the Model):

from sklearn.linear_model import Ridge

model = Ridge(alpha=0.5)
model.fit(X_train_poly, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

🔧 How to Run the Project?

Option 1: Run Locally

Open index.html in a browser.

Enter the required values.

Click "Predict Weight Loss" to get results.

Option 2: Use a Local Server (Recommended)

Install Live Server in VS Code.

Open the project folder in VS Code.

Right-click index.html → "Open with Live Server".

Access the site at http://localhost:5500/.

🌐 Want to Host It Online?

You can host the project on:

GitHub Pages (Best for static sites)

Netlify (Simple drag & drop hosting)

Vercel (Fast & free hosting)

🛠️ Future Improvements

✅ Add Graphs & Visualizations using Chart.js

✅ Improve UI with better mobile responsiveness

✅ Use a Neural Network model for better predictions

🚀 Hope you find this project useful! Feel free to contribute. 😊

