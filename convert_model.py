from keras.models import load_model

# Path to your old model
model = load_model("models/mobilenet_model.h5")

# Save the model in the modern '.keras' format
model.save("models/fish_model_converted.keras", save_format="keras")

print("✅ Model successfully converted to .keras format.")
