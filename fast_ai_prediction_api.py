import joblib
import pandas as pd
import json

class FastAIModel:
    """
    Lightning-fast ML Engine for production backend API usage.
    It instantly loads the compiled '.pkl' file, eliminating all training lag.
    """
    def __init__(self, pkl_path: str = "furniture_ai_master.pkl"):
        print(f"Loading '{pkl_path}' into memory... (Instant Inference Mode)")
        self.brain = joblib.load(pkl_path)
        
        self.models = self.brain['models']
        self.encoders = self.brain['encoders']
        self.features_cat = self.brain['features_cat']
        self.features_num = self.brain['features_num']
        self.targets_num = self.brain['targets_num']
        self.targets_cat = self.brain['targets_cat']
        
    def _encode_for_inference(self, value, col_name):
        le = self.encoders[col_name]
        val_str = str(value)
        if val_str not in le.classes_:
            val_str = le.classes_[0]
        return le.transform([val_str])[0]
        
    def predict_json(self, user_answers: dict) -> dict:
        input_data = {
            'free_space_cm': user_answers.get('free_space_cm', 150),
            'wall_color': user_answers.get('wall_color', 'white'),
            'room_style': user_answers.get('room_style', 'modern'),
            'is_behind_door': user_answers.get('is_behind_door', False),
            'item_type': user_answers.get('item_type', 'wallStorage'),
            'room_type': user_answers.get('room_type', 'bedroom'),
            'floor_material': user_answers.get('floor_material', 'wood'),
            'light_level': user_answers.get('light_level', 'medium'),
            'age_group': user_answers.get('age_group', 'adult'),
            'has_pets': user_answers.get('has_pets', False),
            'existing_material': user_answers.get('existing_material', 'wood')
        }
        
        item_type = input_data['item_type'].lower()
        
        # Instantly compile the math row for the AI
        X_test = pd.DataFrame()
        X_test['free_space_cm'] = [input_data['free_space_cm']]
        for col in self.features_cat:
            X_test[col + '_enc'] = [self._encode_for_inference(input_data[col], col)]
            
        generated_item = {}
        
        # 1. Predict all Continuous Numbers
        for col in self.targets_num:
            val = self.models[col].predict(X_test)[0]
            if col == 'drawersCount':
                generated_item[col] = max(0, int(round(val)))
            else:
                generated_item[col] = int(round(val)) # The required schema uses rounded ints
                
        # 2. Predict all Categorical Variables
        for col in self.targets_cat:
            if col == 'recommended_style': continue
                
            enc_val = self.models[col].predict(X_test)[0]
            val_str = self.encoders['target_' + col].inverse_transform([enc_val])[0]
            if val_str.lower() == 'true': generated_item[col] = True
            elif val_str.lower() == 'false': generated_item[col] = False
            else: generated_item[col] = val_str
            
        # Hard spatial constraint block
        if generated_item['widthCm'] > input_data['free_space_cm']:
             generated_item['widthCm'] = int(input_data['free_space_cm'])
             
        # User requested to hardcode the output style to exactly match the input room style
        generated_item['style'] = input_data['room_style']

        # -----------------------------------------------------------------
        # NEW JSON SCHEMA FORMATTING logic as requested by user
        # -----------------------------------------------------------------
        
        # Calculate rowConfigs intelligently based on predicted height & item type
        row_configs = []
        if 'desk' not in item_type: # Desks typically have no rows
            # Taller items like wallStorage have more rows
            num_rows = max(1, generated_item['heightCm'] // 40)
            if num_rows > 4: num_rows = 4  # Cap to 4 rows to match schema design limits
            
            drawers_remaining = generated_item.get('drawersCount', 0)
            
            for i in range(num_rows):
                # Distribute drawers and doors dynamically
                doors = "none"
                drawers = "none"
                
                # Bottom rows usually get drawers
                if i >= num_rows - 2 and drawers_remaining > 0:
                    drawers = "some" if drawers_remaining < 2 else "all"
                    drawers_remaining -= 2
                
                # Top rows usually get doors 
                if i < 2 and generated_item['heightCm'] > 100:
                    doors = "some" if i == 1 else "all"
                    
                height_size = "lg" if i == num_rows -1 else ("sm" if i == 0 else "md")
                
                row_configs.append({
                    "height": height_size,
                    "doors": doors,
                    "drawers": drawers
                })

        # Assemble the Exact Ordered Schema Required By The New JSON
        final_schema_payload = {
            "widthCm": generated_item['widthCm'],
            "heightCm": generated_item['heightCm'],
            "depthCm": generated_item['depthCm'],
            "color": generated_item['color'],
            "style": generated_item['style'],
            "density": 50, # Arbitrary default visual density integer requested
            "withBack": True, # Hard default boolean requested
            "topStorage": {"height": "md"} if generated_item['heightCm'] > 120 and 'desk' not in item_type else None,
            "bottomStorage": None,
            "rowConfigs": row_configs
        }

        return final_schema_payload

if __name__ == "__main__":
    print("==================================================")
    print("   LIGHTNING-FAST INFERENCE API TESTING           ")
    print("==================================================")
    
    api_endpoint = FastAIModel("furniture_ai_master.pkl")
    
    sample_frontend_request = {
      "free_space_cm": 150,
      "wall_color": "charcoal",
      "room_style": "minimal",
      "is_behind_door": False,
      "item_type": "wallStorage" # Specifically testing wall storage to see array logic
    }
    
    output_payload = api_endpoint.predict_json(sample_frontend_request)
    
    print("\n--- Final Schema Generation Response ---")
    print(json.dumps(output_payload, indent=2))
