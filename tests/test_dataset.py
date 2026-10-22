"""On-disk exchange contract: chronology, held targets and modality isolation."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_io import EpisodeWriter, validate_episode


class DatasetTests(unittest.TestCase):
    def record(self,root):
        writer=EpisodeWriter(root,dict(cameras={'overview':{'quaternion_world_wxyz':[1,0,0,0]}},image_shape=[16,16,3]))
        self.assertEqual(json.loads((root/'attempt.json').read_text())['status'],'in_progress')
        rgb=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
        for i in range(3):
            frame=SimpleNamespace(t_s=3+i*.05,q=np.full((2,6),i),qd=np.zeros((2,6)),
                cube=np.arange(13),pads=np.ones((2,2)),environment=np.zeros(6))
            action=np.full(12,i+.1,dtype=np.float32)
            writer.begin(frame,{'overview':rgb},action)
            for _ in range(3):writer.applied(action)
        writer.finish(True,{'success':True},frame)
        return frame

    def test_alignment_and_privileged_separation_on_real_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'attempt';self.record(root)
            self.assertEqual(validate_episode(root)['frames'],3)
            with np.load(root/'trajectory.npz') as data:
                self.assertEqual(set(data.files),{'state','action','time_s','physics_step','image_step','action_60hz'})
                np.testing.assert_array_equal(data['state'][:,0],[0,1,2])
                np.testing.assert_allclose(data['action'][:,0],[.1,1.1,2.1])
            self.assertEqual(len(json.loads((root/'privileged.json').read_text())),3)
            self.assertEqual(json.loads((root/'attempt.json').read_text())['status'],'completed')
            with self.assertRaises(FileExistsError):EpisodeWriter(root,{})

    def test_nonheld_action_and_missing_image_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'attempt';self.record(root)
            with np.load(root/'trajectory.npz') as data:rows={k:data[k] for k in data.files}
            rows['action_60hz'][1,2,4]+=.01
            np.savez_compressed(root/'trajectory.npz',**rows)
            with self.assertRaisesRegex(ValueError,'actually held'):validate_episode(root)
            rows['action_60hz']=np.repeat(rows['action'][:,None],3,axis=1)
            np.savez_compressed(root/'trajectory.npz',**rows)
            (root/'overview'/'000001.png').unlink()
            with self.assertRaisesRegex(ValueError,'Missing RGB'):validate_episode(root)

    def test_partial_terminal_transition_is_not_training_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'attempt';terminal=self.record(root)
            # A second attempt fails after one actual substep.
            failed=EpisodeWriter(Path(temp)/'failed',dict(cameras={'overview':{}},image_shape=[16,16,3]))
            failed.begin(terminal,{'overview':np.zeros((16,16,3),dtype=np.uint8)},np.zeros(12))
            failed.applied(np.zeros(12));failed.finish(False,{'success':False},terminal)
            meta=json.loads((failed.root/'episode.json').read_text())
            self.assertEqual(meta['frames'],0)
            self.assertEqual(meta['incomplete_terminal_ticks'],1)
            self.assertFalse((failed.root/'trajectory.npz').exists())


if __name__=='__main__':unittest.main()
