import unittest
import torch
from src.models.distillation import distillation_loss


class DistillationTests(unittest.TestCase):
    def test_gradient_only_updates_student(self):
        student = torch.tensor([[2., -1., 0.]], requires_grad=True)
        teacher = torch.tensor([[0., 2., -1.]], requires_grad=True)
        loss = distillation_loss(student, torch.tensor([1]), teacher)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)

    def test_equal_distributions_have_zero_soft_loss(self):
        logits = torch.tensor([[1., 2., 3.]])
        loss = distillation_loss(logits, torch.tensor([2]), logits, alpha=1.0)
        self.assertAlmostEqual(loss.item(), 0, places=6)


if __name__ == "__main__":
    unittest.main()
