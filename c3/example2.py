# [Simple dataset with one feature (e.g., hours studied) and one target (exam score), illustrating best fit line]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Create a small dataset
#    hours_studied: Single feature
#    exam_score: Target variable
hours_studied = np.array([1, 2, 3, 4, 5, 6])
exam_score = np.array([50, 60, 65, 70, 75, 90])

# 2. Reshape the feature array to be 2D (as required by scikit-learn)
X = hours_studied.reshape(-1, 1)
y = exam_score

# 3. Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# 4. Retrieve model parameters (coefficient and intercept)
slope = model.coef_[0]  # Since there's only one feature, we have just one coefficient
intercept = model.intercept_
print(f"Slope (Coefficient): {slope:.3f}")
print(f"Intercept: {intercept:.3f}")

# 5. Predict scores for the training data
y_pred = model.predict(X)

# 6. Plot the data points and the best fit line
plt.scatter(hours_studied, exam_score, label="Data Points")
plt.plot(hours_studied, y_pred, label="Best Fit Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Hours Studied vs Exam Score")
plt.legend()
plt.show()
