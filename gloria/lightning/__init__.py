from .pretrain_model import PretrainModel
from .classification_model import ClassificationModel

LIGHTNING_MODULES = {
    'pretrain': PretrainModel,
    'classification': ClassificationModel}