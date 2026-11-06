import matplotlib.pyplot as plt

# Calculate model accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Model comparison
models = ["Logistic Regression", "Decision Tree"]
accuracies = [lr_accuracy, dt_accuracy]

# Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.title("Student Performance Prediction - Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)

plt.show()
