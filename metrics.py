import torch
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall
from typing import Tuple

class MulticlassMetricsCalculator:
    def __init__(self, num_classes: int, avg_method: str = "micro", device=None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        assert avg_method in ["micro", "macro", "None", None], f'avg method should be in ["micro", "macro", "None", None], found {avg_method}'

        if avg_method == "None":
            avg_method = None

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average=avg_method).to(device)
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average=avg_method).to(device)
        self.auprc = MulticlassAUPRC(num_classes=num_classes).to(device)
        self.auroc = MulticlassAUROC(num_classes=num_classes).to(device)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average=avg_method).to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average=avg_method).to(device)
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        outputs = outputs.to(torch.float32)
        targets = targets.to(torch.long)

        self.accuracy.update(outputs, targets)
        self.f1_score.update(outputs, targets)
        self.auprc.update(outputs, targets)
        self.auroc.update(outputs, targets)
        self.precision.update(outputs, targets)
        self.recall.update(outputs, targets)
        self.confusion_matrix.update(outputs, targets)

    def compute(self) -> Tuple:
        accuracy = self.accuracy.compute()
        f1_score = self.f1_score.compute()
        auprc = self.auprc.compute()
        auroc = self.auroc.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        confusion_matrix = self.confusion_matrix.compute()
        return accuracy, f1_score, auprc, auroc, precision, recall, confusion_matrix
    
    def reset(self) -> None:
        self.accuracy.reset()
        self.f1_score.reset()
        self.auprc.reset()
        self.auroc.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()