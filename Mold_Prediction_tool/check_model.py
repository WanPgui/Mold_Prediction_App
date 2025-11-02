from tensorflow.keras.models import load_model

MODEL_PATH = r"C:\Users\HP\Downloads\Mold_Prediction_tool\models\mold_model_final.keras"

try:
    m = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
    print("Input shape:", m.input_shape)
    print("Output shape:", m.output_shape)
except Exception as e:
    print("Error loading model:", e)
