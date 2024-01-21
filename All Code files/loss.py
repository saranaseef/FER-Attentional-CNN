import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, class_num, param=0.0):
        super(CustomLoss, self).__init__()
        self.class_num = class_num
        self.th = 1 / self.class_num
        self.param = param

    def forward(self, y_pred, y_true2):
        self.batch_size = y_pred.size(0)
        self.cp_label = y_true2[self.batch_size:]
        y_true = y_true2[:self.batch_size]

        y_true = y_true.view(self.batch_size, 1)
        self.cp_label = self.cp_label.view(self.batch_size, 1)

        cp_label_onehot = F.one_hot(self.cp_label, self.class_num).float()
        y_true_onehot = F.one_hot(y_true, self.class_num).float()
        predict_label = torch.argmax(y_pred, dim=1).view(self.batch_size, 1)

        NL_score = self.NL(y_pred, self.cp_label)
        PL_score = self.PL(y_pred, predict_label)
        score = NL_score + self.param * PL_score

        return score

    def NL(self, y_pred, cp_label):
        index = torch.arange(self.batch_size).view(self.batch_size, 1)
        index = torch.cat((index, cp_label), dim=1)
        py = y_pred.gather(1, index)

        cp_label_onehot = F.one_hot(cp_label, self.class_num).float()
        cross_entropy = cp_label_onehot * torch.log(torch.clamp(1 - y_pred, 1e-3, 1.0))
        cross_entropy = torch.sum(cross_entropy, dim=1)
        weight = -(1 - py)
        out = -torch.sum(weight * cross_entropy, dim=1)
        return out / float(self.batch_size)

    def PL(self, y_pred, pred_label):
        one = torch.ones_like(y_pred)
        zero = torch.zeros_like(y_pred)
        label = torch.where(y_pred < self.th, zero, one)
        label = torch.sum(label, dim=1)
        label = torch.where(label < 2, one, zero)
        D = y_pred * label.view(self.batch_size, 1)

        num = self.batch_size
        index = torch.arange(num).view(num, 1)
        D_label = torch.argmax(D, dim=1).view(num, 1)
        index = torch.cat((index, D_label), dim=1)
        py = D.gather(1, index)
        py = 1 - torch.square(py)
        py = py.view(num, 1)
        weight = torch.prod(py, dim=0)

        one_hot = self.y_true_onehot * label.view(self.batch_size, 1)
        cross_entropy = one_hot * torch.log(y_pred)
        cross_entropy = torch.sum(cross_entropy, dim=1)
        out = -weight * cross_entropy
        return torch.sum(out)


