from flask import Flask, request, jsonify
from fast_ai_prediction_api import FastAIModel
from flask_cors import CORS
import traceback

app = Flask(__name__)
# Enable CORS so your frontend can directly make requests to the API
CORS(app)

# Load the AI Brain into memory ONE TIME when the server starts
try:
    print("Booting up ML Engine...")
    ai_engine = FastAIModel("furniture_ai_master.pkl")
    print("ML Engine Ready! Server is actively listening.")
except Exception as e:
    print(f"Critical Error loading .pkl: {e}")
    ai_engine = None

@app.route('/api/predict', methods=['POST'])
def generate_furniture():
    """
    POST Endpoint that expects ONLY the room answers and generates ONE optimal JSON config.
    Example POST Body:
    {
      "room_answers": {
         "free_space_cm": 110,
         "wall_color": "charcoal",
         "room_style": "minimal",
         "is_behind_door": true,
         "item_type": "wallStorage"
      }
    }
    """
    if not ai_engine:
        return jsonify({"error": "AI Model failed to load on server boot."}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400
            
        # Extract the single section from the frontend payload
        room_answers = data.get("room_answers", {})
        
        # Guard clause for empty objects
        if not room_answers:
            return jsonify({"error": "Missing 'room_answers' block in payload."}), 400
            
        # Run the blazing fast ML pipeline to get exactly ONE result
        final_prediction = ai_engine.predict_json(room_answers)
        
        # Return cleanly to the frontend
        return jsonify({
            "status": "success",
            "message": "Optimal ML structure generated successfully.",
            "data": final_prediction
        }), 200
        
    except Exception as e:
        print(f"Server Error during prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the server on Port 5001 (accessible locally)

    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)