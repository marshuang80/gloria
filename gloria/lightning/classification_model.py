import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json
import os
import copy

from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from .. import builder
from .. import gloria
from pytorch_lightning.core import LightningModule


class ClassificationModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg

        if self.cfg.model.vision.model_name in gloria.available_models():
            self.model = gloria.load_img_classification_model(
                self.cfg.model.vision.model_name,
                num_cls=self.cfg.model.vision.num_targets,
                freeze_encoder=self.cfg.model.vision.freeze_cnn,
                pretrained=self.cfg.model.vision.pretrained,
            )
        else:
            self.model = builder.build_img_model(cfg)

        self.loss = builder.build_loss(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split):
        """Similar to traning step"""

        x, y = batch

        logit = self.model(x)
        loss = self.loss(logit, y)

        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "logit": logit, "y": y}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        logit = torch.cat([x["logit"] for x in step_outputs])
        y = torch.cat([x["y"] for x in step_outputs])
        prob = torch.sigmoid(logit)

        y = y.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()

        macro_auroc_list, macro_auprc_list, macro_f1_list = [], [], []
        all_task_label, all_y_cls, all_prob_cls, all_pred_cls = [], [], [], []
        for i in range(y.shape[1]):
            y_cls = y[:, i]
            prob_cls = prob[:, i]
            pred_cls = (prob_cls > 0.5).astype(int)

            # print(split, sum(y_cls), len(y_cls))
            if np.isnan(prob_cls).any():
                macro_auroc_list.append(0)
                macro_auprc_list.append(0)
                auroc_list.append(0)
            else:
                try:
                    macro_auroc_for_task = roc_auc_score(y_cls, prob_cls)
                except:
                    macro_auroc_for_task = 0
                macro_auroc_list.append(macro_auroc_for_task)
                macro_auprc_list.append(average_precision_score(y_cls, prob_cls))
                macro_f1_list.append(f1_score(y_cls, pred_cls))

                all_task_label.extend(np.zeros_like(y_cls) + i)
                all_y_cls.extend(y_cls)
                all_prob_cls.extend(prob_cls)
                all_pred_cls.extend(pred_cls)

        macro_auroc = np.mean(macro_auroc_list)
        macro_auprc = np.mean(macro_auprc_list)
        macro_f1 = np.mean(macro_f1_list)

        try:
            micro_auroc = roc_auc_score(all_y_cls, all_prob_cls)
        except:
            micro_auroc = 0
        micro_auprc = average_precision_score(all_y_cls, all_prob_cls)
        micro_f1 = f1_score(all_y_cls, all_pred_cls)

        self.log(f"{split}_macro_auroc", macro_auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_macro_auprc", macro_auprc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_macro_f1", macro_f1, on_epoch=True, logger=True, prog_bar=True)

        self.log(f"{split}_micro_auroc", micro_auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_micro_auprc", micro_auprc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_micro_f1", micro_f1, on_epoch=True, logger=True, prog_bar=True)

        if split == "test":
            results_csv = os.path.join(self.cfg.output_dir, "results.csv")
            results = {
                "macro_auroc": macro_auroc, 
                "macro_auprc": macro_auprc,
                "macro_f1": macro_f1,
                "micro_auroc": micro_auroc, 
                "micro_auprc": micro_auprc,
                "micro_f1": micro_f1,
            }
            with open(results_csv, "w") as fp:
                json.dump(results, fp)

            output_df = pd.DataFrame(list(zip(all_task_label, all_y_cls, all_prob_cls, all_pred_cls)), columns=['task', 'true', 'prob', 'pred'])
            output_df.to_csv(os.path.join(self.cfg.output_dir, "outputs.csv"), index=False)
