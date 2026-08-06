import logging
import fastertransformer as ft
from djl_python import Input, Output

# Use standard Python logging instead of custom logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model = None


def load_model(properties):
    model_name = properties["model_id"]
    tensor_parallel_degree = int(properties["tensor_parallel_degree"])
    pipeline_parallel_degree = 1
    dtype = properties["dtype"]

    logger.info(f"Loading model: {model_name}")
    # Initialilizing model with FasterTransformer
    model = ft.init_inference(model_name, tensor_parallel_degree, pipeline_parallel_degree, dtype)
    return model


def handle(inputs: Input):
    global model

    if not model:
        model = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    logger.info(f"data: {data}")
    input_text = data["text"]
    params = data["parameters"]
    result = model.pipeline_generate(input_text, **params)
    logger.info(f"result: {result}")

    return Output().add(result)
