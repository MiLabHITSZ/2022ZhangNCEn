import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, model_list, model_name, ensemble_type):
        super(EnsembleModel, self).__init__()
        self.model_list = model_list
        self.ensemble_type = ensemble_type
        self.mode_name = model_name

    def forward(self, feature):
        output_list = []
        logits_list = []

        for model in self.model_list:
            output_list.append(model(feature))
            logits_list.append(model(feature))

        ensemble_output = self.ensemble(output_list, logits_list)
        return ensemble_output

    def ensemble(self, output_list, logits_list=None):
        if self.ensemble_type == 'predict_mean':
            output_mean = torch.mean(torch.stack(output_list, dim=0), dim=0)
            return torch.softmax(output_mean, dim=-1)
        elif self.ensemble_type == 'logtis_mean':
            return torch.mean(logits_list)


