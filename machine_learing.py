from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Training data
training_hours = [[3], [4], [5], [6], [7], [8], [9], [10], [11]]

performance_scores = [60, 65, 70, 76, 85, 83, 88, 92, 95]

# Create the model
model = LinearRegression()

# Train the model
model.fit(training_hours, performance_scores)

predicted_scores = model.predict(training_hours)

accuracy = r2_score(performance_scores, predicted_scores)

print("Model R² Score:", round(accuracy, 2))

# Make a prediction
new_hours = [[12]]
prediction = model.predict(new_hours)

print("Training Hours:", new_hours[0][0])
print("Predicted Performance Score:", round(prediction[0], 2))

new_hours = [[5], [8], [10], [12]]

predictions = model.predict(new_hours)

print("\nPredictions:")

for hours, score in zip(new_hours, predictions):
    print(
        "Training Hours:", hours[0],
        "-> Predicted Performance:", round(score, 2)
    )

    
hours = float(input("\nEnter training hours: "))

user_prediction = model.predict([[hours]])

print(
    "Predicted Performance Score:",
    round(user_prediction[0], 2)
)

plt.scatter(training_hours, performance_scores, label="Actual Data")

plt.plot(
    training_hours,
    predicted_scores,
    label="Prediction"
)

plt.xlabel("Training Hours")
plt.ylabel("Performance Score")
plt.title("Machine Learning: Training Hours vs Performance")

plt.legend()
plt.show()