"""CLI defaults and authorization boundaries; synthetic files are never demonstrations."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import so101
import policy_pipeline
from policy_cameras import CALIBRATION_SHA256


class CommandTests(unittest.TestCase):
    def test_help_is_lightweight_for_every_operation(self):
        for arguments in ([], ['scene'], ['expert'], ['collect'], ['collect', 'run'],
                          ['dataset'], ['dataset', 'convert'], ['train'], ['evaluate', 'serve'], ['replay']):
            code = "import sys;sys.path.insert(0,'scripts');import so101\ntry:so101.parser().parse_args(" + repr(arguments + ['--help']) + ")\nexcept SystemExit as e:assert e.code==0\nassert not any(x in sys.modules for x in ['torch','numpy','isaaclab','lerobot'])"
            result = subprocess.run([sys.executable, '-c', code], cwd=so101.REPO, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_expert_defaults_are_the_validated_recipe_and_cannot_record(self):
        args = so101.parser().parse_args(['expert', '--dry-run'])
        command = so101.controller_command(args)
        for option, value in [('--stage', 'full'), ('--motion-mode', 'continuous'), ('--expert-hz', '20')]:
            self.assertEqual(command[command.index(option)+1], value)
        self.assertIn('--grip-feedback-from-lift', command)
        self.assertNotIn('--headless', command)
        self.assertFalse(any('record' in value for value in command))
        with self.assertRaises(SystemExit), patch('sys.stderr', new=io.StringIO()):
            so101.parser().parse_args(['expert', '--record-dir', '/tmp/not-authorized'])

    def test_collection_cannot_mislabel_a_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(partition='heldout')))
            args = so101.parser().parse_args(['collect', 'run', '--split', 'train', '--manifest', str(path),
                '--raw', directory+'/raw', '--output', directory+'/run', '--execute'])
            with patch('so101.launch') as launch, self.assertRaisesRegex(ValueError, 'partition'):
                so101.dispatch(args)
            launch.assert_not_called()

    def test_evaluation_cannot_call_training_cases_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'manifest.json'
            path.write_text(json.dumps(dict(partition='train',split='train')))
            args = so101.parser().parse_args(['evaluate', 'run', '--split', 'heldout', '--manifest', str(path),
                '--socket', directory+'/absent.sock', '--output', directory+'/evaluation', '--dry-run'])
            with patch('so101.launch') as launch, self.assertRaisesRegex(ValueError, 'never training'):
                so101.dispatch(args)
            launch.assert_not_called()

    def test_replay_preview_preserves_data_and_uses_original_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'episode';root.mkdir()
            metadata = dict(case='a_corner', configuration=[[-.35,-.052],-6.,123], split='validation')
            path = root/'episode.json';path.write_text(json.dumps(metadata))
            before = path.read_bytes()
            args = so101.parser().parse_args(['replay', '--episode', str(root), '--output', directory+'/replay', '--dry-run'])
            with patch('so101.launch', return_value=0) as launch:
                so101.dispatch(args)
            command = launch.call_args.args[0]
            self.assertIn('--replay', command)
            self.assertIn('--cases-file', command)
            self.assertNotIn('--record-dir', command)
            self.assertFalse(args.output.exists())
            self.assertEqual(path.read_bytes(), before)

    def test_inventory_retains_failures_and_unfinished_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)/'raw';raw.mkdir()
            complete = raw/'failed';complete.mkdir();(raw/'interrupted').mkdir()
            (complete/'episode.json').write_text(json.dumps(dict(case='failed',split='train',evaluation={'success':False})))
            ledger = Path(directory)/'attempts.jsonl'
            ledger.write_text('\n'.join(json.dumps(dict(event='started',case=name)) for name in ['failed','interrupted','no_raw']))
            output = io.StringIO()
            with redirect_stdout(output):
                so101.inspect_dataset(SimpleNamespace(raw=raw,quality_report=None,ledger=ledger))
            report = json.loads(output.getvalue())
            self.assertEqual(report['raw_attempt_directories'], 2)
            self.assertEqual(report['ledger_started_attempts'], 3)
            self.assertFalse(any(row['physical_success'] for row in report['episodes']))
            self.assertTrue(all(row['automatic_qa'] is None for row in report['episodes']))


class TrainingGates(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root/'meta').mkdir()
        features = {f'observation.images.{name}': {'shape':[480,640,3]} for name in ['overview','left_wrist','right_wrist']}
        features.update({'action':{'shape':[12]},'observation.state':{'shape':[24]}})
        self.info = dict(fps=20,total_episodes=1,features=features)
        self.conversion = dict(split='train',accepted_episodes=[{'raw':'synthetic-only'}],
            accepted_quality_report=dict(partition='train',ready_for_conversion=True,ledger_complete=True,
                camera_config_sha256=CALIBRATION_SHA256,
                episodes=[dict(path='synthetic-only',accepted=True,automatic_pass=True,digest='synthetic')]))
        self.write_metadata()
        self.args = SimpleNamespace(config='matched-v1',policy='act',dataset=self.root,
            output=self.root/'new-output',seed=17,resume=None,execute=False,authorize_training=False,approval=None)

    def tearDown(self):
        self.temporary.cleanup()

    def write_metadata(self):
        (self.root/'meta/info.json').write_text(json.dumps(self.info))
        (self.root/'conversion.json').write_text(json.dumps(self.conversion))

    def test_pilot_and_heldout_are_never_training_data(self):
        for split in ['pilot','validation','heldout']:
            self.conversion['split'] = split;self.write_metadata()
            with self.assertRaisesRegex(ValueError, 'train conversion'):
                policy_pipeline.training_dataset_proof(self.root)

    def test_automatic_qa_does_not_replace_human_acceptance(self):
        self.conversion['accepted_quality_report']['episodes'][0]['accepted'] = False
        self.write_metadata()
        with self.assertRaises(ValueError):
            policy_pipeline.training_dataset_proof(self.root)

    def test_dry_run_and_missing_authorization_cannot_launch_trainer(self):
        with patch('policy_pipeline.parse_official_training', return_value=SimpleNamespace(policy=SimpleNamespace(type='act',device='cuda'))), \
             patch('subprocess.run') as run, redirect_stdout(io.StringIO()):
            policy_pipeline.train_official(self.args)
            run.assert_not_called()
            self.args.execute = True
            with self.assertRaisesRegex(ValueError, 'authorization'):
                policy_pipeline.train_official(self.args)
            run.assert_not_called()
            self.args.authorize_training = True
            self.args.approval = self.root/'approval.json'
            self.args.approval.write_text(json.dumps(dict(approved_training=True,reviewer='fixture',date='synthetic')))
            with self.assertRaisesRegex(ValueError, 'bind'):
                policy_pipeline.train_official(self.args)
            run.assert_not_called()

    def test_recipe_uses_official_trainer_and_matched_horizons(self):
        for policy in ['act','diffusion']:
            self.args.policy = policy
            command = policy_pipeline.training_command(self.args)
            self.assertEqual(command[1:3], ['-m','lerobot.scripts.lerobot_train'])
            self.assertIn('--policy.n_obs_steps=1', command)
            self.assertIn('--policy.n_action_steps=8', command)
            self.assertIn('--dataset.eval_split=0.0', command)
            self.assertIn('--policy.push_to_hub=false', command)


if __name__ == '__main__':
    unittest.main()
