import torch.nn.functional as F
from transformers import Trainer
import torch
import numpy as np
from torch import nn

class OLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels

        dist_matrix = np.zeros((num_classes, num_classes))

        for i in range(num_classes):
            for j in range(num_classes):
                dist_matrix[i][j] = np.abs(i-j)

        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits,dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances,device='cuda:0', requires_grad=True)
        err = -torch.log(1-probas)*abs(distances_tensor)**2
        loss = torch.sum(err,axis=1).mean()
        return (loss, outputs) if return_outputs else loss

class WKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        num_classes = model.module.num_labels
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        y_pred = F.softmax(logits,dim=1)
        label_vec = torch.range(0,num_classes-1, dtype=torch.float)
        row_label_vec = torch.tensor(torch.reshape(label_vec, (1, num_classes)), requires_grad=True,device='cuda:0')
        col_label_vec = torch.tensor(torch.reshape(label_vec, (num_classes, 1)), requires_grad=True,device='cuda:0')
        col_mat = torch.tile(col_label_vec, (1, num_classes))
        row_mat = torch.tile(row_label_vec, (num_classes, 1))
        weight_mat = (col_mat - row_mat) ** 2
        y_true = torch.tensor(F.one_hot(labels, num_classes=num_classes), dtype=col_label_vec.dtype, requires_grad=True)
        batch_size = y_true.shape[0]
        cat_labels = torch.matmul(y_true, col_label_vec)
        cat_label_mat = torch.tensor(torch.tile(cat_labels, [1, num_classes]), requires_grad=True,device='cuda:0')
        row_label_mat = torch.tensor(torch.tile(row_label_vec, [batch_size, 1]), requires_grad=True,device='cuda:0')
        
        weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, axis=0)
        pred_dist = torch.sum(y_pred, axis=0)
        w_pred_dist = torch.t(torch.matmul(weight_mat, pred_dist))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist/batch_size),axis = 0)
        loss = torch.log(numerator/denominator + 1e-7)
 
        return (loss, outputs) if return_outputs else loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class TrainerwithFocalLoss(Trainer):
    #Source https://github.com/boostcampaitech4lv23nlp1/level2_klue_nlp-level2-nlp-07/blob/bafe473086440f12578e24cffdc14003b58ee953/code/utils.py#L87
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(inputs)
        logits = outputs['output']
        
        loss_fct = FocalLoss()
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss