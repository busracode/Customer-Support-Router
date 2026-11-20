"""Öğretmen olasılıkları ve gerçek etiketlerle bilgi damıtma kaybı."""
import torch.nn.functional as F


def distillation_loss(student_logits, labels, teacher_logits, temperature=2.0, alpha=0.5):
    hard_loss = F.cross_entropy(student_logits.float(), labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=-1),
        F.softmax(teacher_logits.detach().float() / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature ** 2
    return (1 - alpha) * hard_loss + alpha * soft_loss
