import torch
import torch.nn.functional as functional


class CentroidLossV2(torch.nn.Module):
    @staticmethod
    def vector_loss(result: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ret = torch.mean(torch.sqrt(torch.sum((target - result) ** 2, dim=1)))
        return ret

    @staticmethod
    def ce_loss(result: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = target.shape[1]
        result_logits = result.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
        target_ids = torch.argmax(target, dim=1).long().view(-1, 1).squeeze()
        ret = functional.cross_entropy(result_logits, target_ids)
        return ret

    def forward(self, result, target):
        centroid_vectors_result, centroid_vectors_target = result[:, 0:2], target[:, 0:2]
        border_vectors_result, border_vectors_target = result[:, 2:4], target[:, 2:4]
        logits_result, logits_target = result[:, 4:], target[:, 4:]

        loss = self.vector_loss(centroid_vectors_result, centroid_vectors_target) + \
            self.vector_loss(border_vectors_result, border_vectors_target) + \
            self.ce_loss(logits_result, logits_target)

        return loss
