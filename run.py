import json
import joblib  # For loading the saved model
import pandas as pd

# Load the dataset (you should have the same dataset for the recommendations)
data = pd.read_csv("sustainable_ai_tools_dataset3.csv")

# Define environmental score
data["Environmental Score"] = (data["Energy Efficiency"] + data["Waste Reduction"]) / data["Carbon Footprint"]

# Define features (accuracy, speed, environmental score)
X = data[["Accuracy", "Speed", "Environmental Score"]]
y = data["Environmental Score"]  # Example: You can define y as a composite score based on specific needs

# Function to load model and get recommendations from JSON input
def get_recommendations_from_json(input_json):
    # Load the trained model from the .h5 file
    model = joblib.load("model.h5")
    
    # Parse the input JSON
    data_input = json.loads(input_json)
    category = data_input['category']
    preference = data_input['preference']
    
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

        # Prepare the recommendations in JSON format
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

    # Get recommendations
    recommendations = recommend_tools(category, preference)

    # Convert the recommendations into JSON format for output
    output_json = json.dumps(recommendations, indent=4)
    return output_json


# Example usage: Providing a JSON input and getting recommendations in JSON format
input_json = json.dumps({
    "category": "Sustainable IoT",
    "preference": "mix"
})

# Get recommendations in JSON format
output_json = get_recommendations_from_json(input_json)
print(output_json)
