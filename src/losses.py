import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, gamma=2):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, outputs, targets):
        # Convert logits to probabilities
        outputs = F.softmax(outputs, dim=1)
        num_classes = outputs.size(1)

        dice_scores = torch.zeros(num_classes).to(outputs.device)

        for i in range(num_classes):
            output_i = outputs[:, i, :, :]
            target_i = (targets == i).float()

            # Compute intersection and union
            intersection = (output_i * target_i).sum()
            union = (output_i ** self.gamma).sum() + (target_i ** self.gamma).sum()

            # Compute Dice Score for the class
            dice_scores[i] = (2. * intersection + self.smooth) / (union + self.smooth)

        # Compute Dice Loss as 1 - mean Dice score
        dice_loss = 1. - dice_scores.mean()

        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, smooth=1e-6, gamma=2, contrib_ratio=0.5):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth, gamma=gamma)
        self.contrib_ratio = contrib_ratio

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)
        dice_loss = self.dice_loss(outputs, targets)
        combined_loss = (self.contrib_ratio * ce_loss) + ((1 - self.contrib_ratio) * dice_loss)
        return combined_loss