import unittest
import torch
from torch import nn
from training.train import train_epoch
from training.datasets import HandSealTransform
from PIL import Image


class TransferLearningTests(unittest.TestCase):
    def test_robust_augmentation_leaves_validation_preprocessing_unchanged(self):
        image = Image.new('RGB', (240, 320), (120, 80, 40))
        regular = HandSealTransform(224, [0, 0, 0], [1, 1, 1], False)
        robust = HandSealTransform(224, [0, 0, 0], [1, 1, 1], False, robust_augmentation=True)
        self.assertTrue(torch.equal(regular(image), robust(image)))
        train = HandSealTransform(224, [0, 0, 0], [1, 1, 1], True, robust_augmentation=True)
        value = train(image)
        self.assertEqual(tuple(value.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(value).all())

    def test_frozen_batchnorm_keeps_statistics_while_features_learn(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))
        before = model[1].running_mean.clone()
        weights = model[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        train_epoch(model, [(torch.randn(8, 4), torch.tensor([0, 1] * 4))], optimizer,
                    nn.CrossEntropyLoss(), 'cpu', freeze_batchnorm=True)
        self.assertTrue(torch.equal(before, model[1].running_mean))
        self.assertFalse(torch.equal(weights, model[0].weight))

    def test_frozen_features_keep_weights_and_batchnorm_statistics(self):
        torch.manual_seed(42)
        model = nn.Sequential()
        model.add_module('features', nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))
        model.add_module('classifier', nn.Linear(4, 2))
        for parameter in model.features.parameters():
            parameter.requires_grad_(False)
        before = {key: value.clone() for key, value in model.features.state_dict().items()}
        head_before = model.classifier.weight.detach().clone()
        loader = [(torch.randn(8, 4), torch.tensor([0, 1] * 4))]
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01)
        train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), 'cpu', freeze_backbone=True)
        for key, value in model.features.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertFalse(torch.equal(head_before, model.classifier.weight))
