from tensorflow.keras.models import load_model

MODELS = {
    'model1': 'models/Model1_CNN.h5',          # Custom CNN
    'model2': 'models/Model2_VGG16.h5',        # VGG16
    'model3': 'models/Model3_VGG19.h5',        # VGG19
    'model4': 'models/Model4_ResNet50.h5',     # ResNet50
    'model5': 'models/Model5_MobileNetV2.h5'   # MobileNetV2
}

_loaded_models = {}


def load_selected_model(model_name: str):
    if model_name not in MODELS:
        model_name = 'model1'
    if model_name not in _loaded_models:
        _loaded_models[model_name] = load_model(MODELS[model_name])
    return _loaded_models[model_name]
