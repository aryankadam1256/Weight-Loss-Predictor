 // Learned Parameters
 const coefficients = [0.05750528, -0.29441001, -0.01331863, 1.4067613, 0.94437414, 0.00215475];
 const intercept = 5.288190272693152;

 // Standardization parameters (replace with actual values)
 const means = [29.91182, 39.31875, 0.46250, 50.53125, 54.18125, 0.95625]; // Actual means
 const stdDevs = [7.524399, 11.936610, 0.500157, 22.562261, 19.550570, 0.819164]; // Actual std devs

 // Normalize and predict function
 function predictWeightLoss() {
     // Get user input
     const bmi = parseFloat(document.getElementById("bmi").value);
     const age = parseInt(document.getElementById("age").value);
     const gender = parseInt(document.getElementById("gender").value);
     const exerciseDuration = parseInt(document.getElementById("exerciseDuration").value);
     const numDays = parseInt(document.getElementById("numDays").value);
     const exerciseType = parseInt(document.getElementById("exerciseType").value);

     // Scale the input values based on the means and standard deviations
     const scaledInput = [
         (bmi - means[0]) / stdDevs[0],
         (age - means[1]) / stdDevs[1],
         (gender - means[2]) / stdDevs[2],
         (exerciseDuration - means[3]) / stdDevs[3],
         (numDays - means[4]) / stdDevs[4],
         (exerciseType - means[5]) / stdDevs[5]
     ];

     // Debug: Log the scaled inputs to check if they are reasonable
     console.log("Scaled Input:", scaledInput);

     // Calculate the predicted weight loss using the learned model
     let predictedWeightLoss = intercept;
     for (let i = 0; i < coefficients.length; i++) {
         predictedWeightLoss += coefficients[i] * scaledInput[i];
     }

     // Debug: Log the predicted value before capping
     console.log("Predicted Weight Loss (before capping):", predictedWeightLoss);

     // Cap the predicted weight loss to 0 and 15
     predictedWeightLoss = Math.max(0, Math.min(15, predictedWeightLoss));

     // Output the result
     document.getElementById("predictionOutput").innerText = predictedWeightLoss.toFixed(2);
 }
 