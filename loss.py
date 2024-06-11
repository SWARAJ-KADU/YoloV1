import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = intersection_over_union(prediction[...,21:25], target[...,21:25])
        iou_b2 = intersection_over_union(prediction[...,26:30], target[...,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        _, bestbox = torch.max(ious, dim = 0)
        exists_box = target[...,20].unsqueeze(3) #Iobj_i

        #FOR BOX COORDINATE
        box_pred = exists_box*((1-bestbox)*prediction[..., 21:25] + bestbox*prediction[...,26:30])
        box_tar = exists_box*target[..., 21:25]
        box_pred[...,2:4] = torch.sign(box_pred[...,2:4])*torch.sqrt(torch.abs(box_pred[...,2:4] + 1e-6))
        box_tar[...,2:4] = torch.sqrt(box_tar[...,2:4])

        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_pred, end_dim=-2),
            torch.flatten(box_tar, end_dim=-2)
        )

        #FOR OBJECT LOSS
        pred_box1 = exists_box*((1-bestbox)*prediction[..., 20:21] + bestbox*prediction[..., 25:26])
        obj_loss = self.mse(
            torch.flatten(pred_box1), 
            torch.flatten(exists_box*target[..., 20:21])
        )

        #FOR NO OBJECT LOSS
        not_exists_box = 1 - exists_box
        no_obj_loss = self.mse(
            torch.flatten(not_exists_box*prediction[..., 20:21]),
            torch.flatten(not_exists_box*target[..., 20:21])
        )
        no_obj_loss = self.mse(
            torch.flatten(not_exists_box*prediction[..., 25:26]),
            torch.flatten(not_exists_box*target[..., 20:21])   
        )

        #
        class_loss = self.mse(
            torch.flatten(prediction[..., 0:20], end_dim = -2), 
            torch.flatten(target[..., 0:20], end_dim = -2)
        )

        loss = self.lambda_coord*box_loss + obj_loss + self.lambda_noobj*no_obj_loss + class_loss
        return loss

