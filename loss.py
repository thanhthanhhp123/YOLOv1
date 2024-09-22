import torch.nn.functional as F
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20, lambda_coord = 5, lambda_noobj = 0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    
    def forward(self, preds, targets):
        #preds (tensor) [batch_size, S, S, Bx5+C]  Bx5 -> [x, y, w, h, conf]
        #targets (tensor) [batch_size, S, S, C+5] 5 -> [conf, x, y, w, h]
        obj_mask = targets[..., 4] > 0 # Đánh dấu các cell chứa object theo confidence score

        # Localization loss
        box_pred = preds[obj_mask][..., :5*self.B].view(-1, self.B, 5) # Lấy các thông số của bounding box từ preds
        box_target = targets[obj_mask][..., :5*self.B].view(-1, self.B, 5) # Lấy các thông số của bounding box từ targets

        box_pred[..., 2:4] = torch.sqrt(box_pred[..., 2:4]) # Chuyển w, h về căn bậc 2
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])
        
        coord_loss = self.lambda_coord * self.mse(box_pred[..., :4], box_target[..., :4]) # Tính loss cho các thông số x, y, w, h

        # Confidence loss
        conf_pred = preds[..., 4:5*self.B].view(-1, self.B)
        conf_target = targets[..., 4:5*self.B].view(-1, self.B)

        conf_obj = self.mse(conf_pred[obj_mask], conf_target[obj_mask])
        conf_noobj = self.lambda_noobj * self.mse(conf_pred[~obj_mask], conf_target[~obj_mask])

        conf_loss = conf_obj + conf_noobj

        # Class loss
        class_pred = preds[obj_mask][..., 5*self.B:]
        class_target = targets[obj_mask][..., 5*self.B:]

        class_loss = self.mse(class_pred, class_target)

        return coord_loss + conf_loss + class_loss
    