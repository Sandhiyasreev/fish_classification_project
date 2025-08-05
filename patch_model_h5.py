import h5py
import json

# Path to your HDF5 model
model_path = "models/mobilenet_model.h5"

# Open the model file
with h5py.File(model_path, "r+") as f:
    print("🔍 Reading model config...")

    # Read config (already a string)
    model_config = json.loads(f.attrs["model_config"])


    # Recursive function to remove 'groups' key
    def remove_groups(config):
        if isinstance(config, dict):
            config.pop("groups", None)
            for value in config.values():
                remove_groups(value)
        elif isinstance(config, list):
            for item in config:
                remove_groups(item)


    # Apply the patch
    remove_groups(model_config)

    # Save updated config back
    f.attrs["model_config"] = json.dumps(model_config)
    print("✅ Successfully patched the model. You can now try loading it again.")

