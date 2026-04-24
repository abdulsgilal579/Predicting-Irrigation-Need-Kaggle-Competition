from sklearn.preprocessing import LabelEncoder

le_exercise = LabelEncoder()

fruits = ['Apple', 'Banana', 'Mango', 'Apple', 'Mango', 'Banana', 'Banana']
encoded = le_exercise.fit_transform(fruits)

# print(f"Original: {fruits}")
# print(f"Encoded: {encoded}")
# print(f"{le_exercise.inverse_transform([0, 1, 2, 0, 2, 1, 1])}")

print(le_exercise.classes_)