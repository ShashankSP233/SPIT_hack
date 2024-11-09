import json
import joblib  # For loading the saved model
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
application = Flask(__name__)

# Load the dataset
data = pd.read_csv("generated_tools_data_sequential.csv")

# Define environmental score with additional environmental parameters
data["Environmental Score"] = (
    (data["Energy Efficiency"] + data["Waste Reduction"] + data["Resource Efficiency"] + data["Lifetime Durability"]) /
    (data["Carbon Footprint"] + data["Energy Consumption"] + data["Water Usage"])
)

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
            alpha, beta, gamma = 0.7, 0.2, 0.1
        elif preference == "speed":
            alpha, beta, gamma = 0.2, 0.7, 0.1
        else:  # mix
            alpha, beta, gamma = 0.4, 0.4, 0.2

        # Calculate combined score with environmental impact
        category_tools["Score"] = (
            alpha * category_tools["Accuracy"] +
            beta * category_tools["Speed"] +
            gamma * category_tools["Environmental Score"]
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
                "Waste Reduction": row["Waste Reduction"],
                "Energy Consumption": row["Energy Consumption"],
                "Resource Efficiency": row["Resource Efficiency"],
                "Water Usage": row["Water Usage"],
                "Lifetime Durability": row["Lifetime Durability"]
            })

        return recommendations

    # Get recommendations
    recommendations = recommend_tools(category, preference)

    # Convert the recommendations into JSON format for output
    output_json = json.dumps(recommendations, indent=4)
    return output_json


@application.route('/recommend', methods=['POST'])
def recommend():
    input_json = request.get_json()
    output_json = get_recommendations_from_json(json.dumps(input_json))
    return jsonify(json.loads(output_json))


if __name__ == "__main__":
    application.run(debug=True)
