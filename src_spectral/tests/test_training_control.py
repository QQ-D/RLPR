import copy
import importlib.util
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.registry import build_params
from trainer.cross_modal import CrossModalTrainer
from utils.seed import set_global_seed


class StochasticSamples(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return torch.tensor([index / 12., random.random(), float(np.random.rand()), float(torch.rand(()))])


class TrainingControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_environment_overrides_stage_default_and_cli_overrides_environment(self):
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': '', 'RLPR_EPOCHS': '10'}):
            self.assertEqual(build_params(1).epochs, 10)
            self.assertEqual(build_params(5).epochs, 10)
            self.assertEqual(build_params(1, epochs_override=3).epochs, 3)

    def test_defaults_are_unchanged_and_invalid_epoch_counts_are_rejected(self):
        with patch.dict(os.environ, {'RLPR_TRAINING_ROOT': ''}):
            os.environ.pop('RLPR_EPOCHS', None)
            self.assertEqual(build_params(1).epochs, 1)
            self.assertEqual(build_params(5).epochs, 5)
            for value in ('0', '-1', '1.5', 'bad'):
                with self.subTest(value=value), patch.dict(os.environ, {'RLPR_EPOCHS': value}):
                    with self.assertRaisesRegex(ValueError, 'epoch'):
                        build_params(1)

    def test_completed_target_is_rejected_instead_of_silently_skipping_training(self):
        trainer = CrossModalTrainer.__new__(CrossModalTrainer)
        trainer.params = SimpleNamespace(epochs=1)
        trainer._init_weights = lambda: 1
        with self.assertRaisesRegex(ValueError, 'epoch'):
            trainer.do_train()

    def test_strict_reproducibility_requires_workspace_and_enables_deterministic_algorithms(self):
        previous = torch.are_deterministic_algorithms_enabled()
        try:
            with patch.dict(os.environ, {'RLPR_DETERMINISTIC': '1', 'CUBLAS_WORKSPACE_CONFIG': ''}):
                with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG'):
                    set_global_seed(0)
            with patch.dict(os.environ, {'RLPR_DETERMINISTIC': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                set_global_seed(0)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous)

    def state_module(self):
        self.assertIsNotNone(importlib.util.find_spec('utils.training_state'),
                             'Reproducible epoch-boundary training state is missing')
        from utils import training_state
        return training_state

    def test_resume_matches_uninterrupted_shuffled_stochastic_training(self):
        state = self.state_module()
        for workers in (0, 1):
            with self.subTest(workers=workers):
                def create():
                    random.seed(11); np.random.seed(11); torch.manual_seed(11)
                    loader = DataLoader(StochasticSamples(), batch_size=4, shuffle=True, num_workers=workers)
                    state.configure_sampling(loader, 0)
                    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.25), torch.nn.Linear(8, 1))
                    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.8)
                    return loader, model, optimizer, scheduler
                def epoch(loader, model, optimizer, scheduler):
                    values = []
                    for x in loader:
                        optimizer.zero_grad()
                        loss = (model(x).flatten() - x[:, 0]).square().mean()
                        loss.backward(); optimizer.step(); values.append(loss.detach().clone())
                    scheduler.step()
                    return torch.stack(values)
                loader, model, optimizer, scheduler = create()
                epoch(loader, model, optimizer, scheduler)
                expected = epoch(loader, model, optimizer, scheduler)
                expected_weights = copy.deepcopy(model.state_dict())
                expected_lr = optimizer.param_groups[0]['lr']
                loader, model, optimizer, scheduler = create()
                epoch(loader, model, optimizer, scheduler)
                checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                                scheduler=scheduler.state_dict(), rng=state.capture_training_state(loader)))
                loader, model, optimizer, scheduler = create()
                model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                random.random(); np.random.rand(); torch.rand(17)
                self.assertTrue(state.restore_training_state(checkpoint['rng'], loader))
                actual = epoch(loader, model, optimizer, scheduler)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for key, value in expected_weights.items():
                    torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
                self.assertEqual(optimizer.param_groups[0]['lr'], expected_lr)

    def test_resume_settings_allow_more_epochs_but_reject_changed_objective(self):
        state = self.state_module()
        checkpoint = {'training_settings': {'alignment_loss': 'signed_coherency_v2', 'batch_size': 8}}
        state.validate_training_settings(checkpoint, {'alignment_loss': 'signed_coherency_v2', 'batch_size': 8})
        with self.assertRaisesRegex(ValueError, 'alignment_loss'):
            state.validate_training_settings(checkpoint, {'alignment_loss': 'infonce', 'batch_size': 8})
        for key, saved, requested in (('alignment_loss', 'infonce', 'signed_coherency_v2'),
                                      ('phase_loss_weight', .1, .5),
                                      ('cross_fusion_loss_weight', .5, .2)):
            with self.subTest(legacy_key=key), self.assertRaisesRegex(ValueError, key):
                state.validate_training_settings({key: saved}, {key: requested})
        self.assertFalse(state.restore_training_state(None, DataLoader(StochasticSamples())))


if __name__ == '__main__':
    unittest.main()
