import joblib

# Load the saved model once
with open("iris_model.pkl", "rb") as f:
    model = joblib.load(f)

# Class names mapping
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Predict a predefined sample
print("Predicting predefined sample:")
sample1 = [[5.1, 3.5, 1.4, 0.2]]
prediction1 = model.predict(sample1)
print(f"Predicted class for {sample1[0]}: {class_names[prediction1[0]]}")

sample2 = [[6.7, 3.1, 4.7, 1.5]]
prediction2 = model.predict(sample2)
print(f"Predicted class for {sample2[0]}: {class_names[prediction2[0]]}")

# Predict using user input
print("\nNow enter your own flower measurements:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

user_sample = [[sepal_length, sepal_width, petal_length, petal_width]]
user_prediction = model.predict(user_sample)

print(f"Predicted Iris class: {class_names[user_prediction[0]]}")
