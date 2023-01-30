from . import data_module
from . import image_dataset
from . import pretraining_dataset

DATA_MODULES = {
    "pretrain": data_module.PretrainingDataModule,
    "chexpert": data_module.CheXpertDataModule,
    "intermountain": data_module.IntermountainDataModule,
    "candid_ptx": data_module.CandidPtxDataModule,
    "pneumothorax": data_module.PneumothoraxDataModule,
    "pneumonia": data_module.PneumoniaDataModule,
}
