function predictWeightLoss() {
    let age = parseFloat(document.getElementById("age").value);
    let weight = parseFloat(document.getElementById("weight").value);
    let height = parseFloat(document.getElementById("height").value);
    let calories = parseFloat(document.getElementById("calories").value);
    let activity = document.getElementById("activity").value;
    let gender = document.getElementById("gender").value;
    let duration = parseFloat(document.getElementById("duration").value);

    if (isNaN(age) || isNaN(weight) || isNaN(height) || isNaN(calories) || isNaN(duration)) {
        alert("Please fill in all fields correctly.");
        return;
    }

    // ML-derived weights and bias
    let W = { 
        age: 0.1091,      
        weight: -0.1148,    
        height: -0.0144,    
        calories: -0.0024,  
        duration: 0.0058,  
        gender: { male: -0.1537, female: 0 },  
        activity: { low: 0, moderate: 0.2284, high: 0.0899 }  
    };
    
    let B = 15.1325;  

    // Convert activity level to numerical value
    let activityEffect = W.activity[activity] || 0;

    // Calculate prediction using the ML formula
    let prediction = (W.age * age) + (W.weight * weight) + (W.height * height) + 
                     (W.calories * calories) + (W.duration * duration) + 
                     W.gender[gender] + activityEffect + B;

    prediction = Math.max(0, prediction); // Ensure no negative predictions

    document.getElementById("result").innerText = prediction.toFixed(2);
}
