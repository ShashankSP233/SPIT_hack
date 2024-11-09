import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv("ai_tool_recommendations.csv")

# Define environmental score
data["Environmental Score"] = (data["Energy Efficiency"] + data["Waste Reduction"]) / data["Carbon Footprint"]

# Define features (accuracy, speed, environmental score)
X = data[["Accuracy", "Speed", "Environmental Score"]]
y = data["Environmental Score"]  # Example: You can define y as a composite score based on specific needs

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline with scaling and regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", DecisionTreeRegressor())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict scores for test data (optional)
y_pred = pipeline.predict(X_test)

# Define recommendation function
def recommend_tools(category, preference, top_n=3):
    # Filter data by category
    category_tools = data[data["Category"] == category]
    
    # Define weights based on preference
    if preference == "accuracy":
        alpha, beta = 0.7, 0.3
    elif preference == "speed":
        alpha, beta = 0.3, 0.7
    else:  # mix
        alpha, beta = 0.5, 0.5

    # Calculate combined score
    category_tools["Score"] = (
        alpha * category_tools["Accuracy"] + 
        beta * category_tools["Speed"] + 
        0.5 * category_tools["Environmental Score"]
    )

    # Sort tools by score and select top_n
    top_recommendations = category_tools.nlargest(top_n, "Score")

    # Display recommendations
    recommendations = []
    for _, row in top_recommendations.iterrows():
        recommendations.append({
            "Tool": row["Tool"],
            "Score": row["Score"],
            "Accuracy": row["Accuracy"],
            "Speed": row["Speed"],
            "Energy Efficiency": row["Energy Efficiency"],
            "Carbon Footprint": row["Carbon Footprint"],
            "Waste Reduction": row["Waste Reduction"]
        })

    return recommendations

# Example usage
user_category = "Category 4"  # Replace with user's input
user_preference = "mix"        # Options: "accuracy", "speed", "mix"

# Get recommendations
recommendations = recommend_tools(user_category, user_preference)

# Print recommendations
print(f"Top 3 recommendations for {user_category} based on '{user_preference}' preference:")
for i, rec in enumerate(recommendations, 1):
    print(f"\nRecommendation {i}:")
    for key, value in rec.items():
        print(f"{key}: {value}")
