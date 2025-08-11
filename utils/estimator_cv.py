import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class EstimatorCV():
    def __init__(self, feature_num, class_num, _device):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.device = _device
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(self.device)
        self.Ave = torch.zeros(class_num, feature_num).to(self.device)
        self.Amount = torch.zeros(class_num).to(self.device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(var_temp.permute(1, 2, 0), var_temp.permute(1, 0, 2)).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0
        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave.detach() - ave_CxA).view(C, A, 1),
                (self.Ave.detach() - ave_CxA).view(C, 1, A)
            )
        )
        self.CoVariance = (copy.deepcopy(self.CoVariance.detach()).mul(1 - weight_CV) + var_temp.mul(
            weight_CV)) + additional_CV

        self.Ave = (copy.deepcopy(self.Ave.detach()).mul(1 - weight_AV) + ave_CxA.mul(weight_AV))
        self.Amount += onehot.sum(0)


class CriterionMCV(nn.Module):
    def __init__(self, feature_num, class_num, device):
        super(CriterionMCV, self).__init__()
        self.estimator = EstimatorCV(feature_num, class_num, device)
        self.cross_entropy = nn.CrossEntropyLoss()


    def forward(self, features, y_s, labels):
        self.estimator.update_CV(features, labels)
        loss = F.cross_entropy(y_s, labels)
        return loss

    def get_cv(self):
        return self.estimator.CoVariance

    def get_average(self):
        return self.estimator.Ave

