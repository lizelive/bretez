from transformers import AutoImageProcessor, AutoModel


SAT_MODEL = "facebook/dinov3-vit7b16-pretrain-sat493m"


class Backbone:
    def __init__(self, model_name=SAT_MODEL):
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state