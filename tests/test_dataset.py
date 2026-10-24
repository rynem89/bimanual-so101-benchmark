"""On-disk exchange contract: chronology, held targets and modality isolation."""
import json
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dataset_io import EpisodeWriter, validate_episode, read_rgb, episode_rgb


class DatasetTests(unittest.TestCase):
    def video_record(self,root,interrupted=False):
        cameras={name:{'quaternion_world_wxyz':[1,0,0,0]} for name in ('overview','left_wrist','right_wrist')}
        writer=EpisodeWriter(root,dict(cameras=cameras,image_shape=[16,16,3],joint_limits_rad=[[-2,3]]*6),image_storage='mp4')
        pattern=np.indices((16,16)).sum(0)%2*40
        for i in range(26):
            frame=SimpleNamespace(t_s=3+i*.05,q=np.zeros((2,6)),qd=np.zeros((2,6)),cube=np.zeros(13),pads=np.ones((2,2)),environment=np.zeros(6))
            rgb=np.repeat((pattern+i*7).astype(np.uint8)[:,:,None],3,axis=2)
            writer.begin(frame,{name:rgb for name in cameras},np.zeros(12,dtype=np.float32))
            for _ in range(1 if i==25 else 3):writer.applied(np.zeros(12,dtype=np.float32))
        writer.finish(not interrupted,{'success':not interrupted},frame,interrupted=interrupted)

    def test_video_exact_counts_seeking_and_terminal_partial_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'video';self.video_record(root)
            meta=json.loads((root/'episode.json').read_text())
            self.assertEqual(validate_episode(root)['frames'],25)
            self.assertEqual(meta['incomplete_terminal_ticks'],1)
            self.assertTrue(meta['terminal_image_recorded'])
            self.assertEqual([v['frames'] for v in meta['videos'].values()],[26]*3)
            frames=list(episode_rgb(root))
            self.assertEqual(len(frames),25)
            for name in meta['cameras']:
                self.assertFalse((root/name).exists())
                for i in (0,12,24):
                    np.testing.assert_array_equal(read_rgb(root,name,i),frames[i][name])
                    self.assertAlmostEqual(float(frames[i][name].mean()),20+i*7,delta=2)
            meta['videos']['overview']['frames']-=1
            (root/'episode.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'frame count'):validate_episode(root)

    def test_interrupted_video_is_finalized_and_never_marked_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'interrupted';self.video_record(root,interrupted=True)
            meta=json.loads((root/'episode.json').read_text())
            self.assertFalse(meta['success']);self.assertTrue(meta['interrupted'])
            self.assertEqual(json.loads((root/'attempt.json').read_text())['status'],'interrupted')
            self.assertEqual(validate_episode(root)['frames'],25)

    def test_html_review_supports_legacy_png_and_three_video_streams(self):
        import so101
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for kind in ('png','mp4'):
                episode=root/kind
                if kind=='png':self.record(episode)
                else:self.video_record(episode)
                output=root/(kind+'.html')
                so101.review_episode(SimpleNamespace(episode=episode,output=output))
                page=output.read_text();payload=json.loads(page.split('const d=',1)[1].split(';const seek',1)[0])
                self.assertEqual(payload['fps'],20)
                self.assertEqual(len(payload['action']),3 if kind=='png' else 25)
                self.assertEqual(len(payload['videos']),0 if kind=='png' else 3)

    @unittest.skipUnless((Path(sys.prefix).parent/'so101-il-py312/bin/python').exists(),'Pinned learning environment unavailable')
    def test_video_rgb_is_accepted_by_pinned_official_lerobot(self):
        # Tiny synthetic storage integration test, never a physical demo or
        # human-accepted corpus. Uses no QA approval, policy or optimizer.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.video_record(root/'raw')
            code="""
import sys,json,numpy as np
from pathlib import Path
from dataset_io import episode_rgb,validate_episode,JOINT_ORDER,TASK
from lerobot.datasets.lerobot_dataset import LeRobotDataset
raw=Path(sys.argv[1]);output=Path(sys.argv[2]);meta=json.loads((raw/'episode.json').read_text())
assert validate_episode(raw)['frames']==25
features={'observation.state':dict(dtype='float32',shape=(24,),names=JOINT_ORDER+[n+'.velocity' for n in JOINT_ORDER]),'action':dict(dtype='float32',shape=(12,),names=JOINT_ORDER)}
features.update({f'observation.images.{name}':dict(dtype='image',shape=(16,16,3),names=['height','width','channels']) for name in meta['cameras']})
dataset=LeRobotDataset.create('local/synthetic_storage_test',fps=20,features=features,root=output,robot_type='bimanual_so101',use_videos=False,image_writer_threads=1,video_backend='pyav')
with np.load(raw/'trajectory.npz') as data:
    for i,images in enumerate(episode_rgb(raw,meta)):
        row={'observation.state':data['state'][i],'action':data['action'][i],'task':TASK}
        row.update({f'observation.images.{name}':rgb for name,rgb in images.items()});dataset.add_frame(row)
    dataset.save_episode()
dataset.finalize()
assert dataset.num_frames==25
print('Pinned official LeRobot accepted 25 three-camera decoded frames')
"""
            result=subprocess.run([str(Path(sys.prefix).parent/'so101-il-py312/bin/python'),'-c',code,str(root/'raw'),str(root/'official')],
                cwd=Path(__file__).resolve().parents[1],env={**__import__('os').environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'scripts')},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def record(self,root,origin=180):
        writer=EpisodeWriter(root,dict(cameras={'overview':{'quaternion_world_wxyz':[1,0,0,0]}},image_shape=[16,16,3],
                                      joint_limits_rad=[[-2,3]]*6))
        self.assertEqual(json.loads((root/'attempt.json').read_text())['status'],'in_progress')
        rgb=np.arange(16*16*3,dtype=np.uint8).reshape(16,16,3)
        for i in range(3):
            frame=SimpleNamespace(t_s=origin/60+i*.05,q=np.full((2,6),i),qd=np.zeros((2,6)),
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

    def test_recording_does_not_require_global_ticks_divisible_by_three(self):
        with tempfile.TemporaryDirectory() as temp:
            for origin in (180,181,182):
                root=Path(temp)/str(origin);self.record(root,origin)
                self.assertEqual(validate_episode(root)['frames'],3)

    def test_degrees_or_corrupt_targets_cannot_pass_as_radians(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'attempt';self.record(root)
            with np.load(root/'trajectory.npz') as data:rows={key:data[key] for key in data.files}
            rows['action'][1,4]=90;rows['action_60hz'][1,:,4]=90
            np.savez_compressed(root/'trajectory.npz',**rows)
            with self.assertRaisesRegex(ValueError,'radian bounds'):validate_episode(root)


if __name__=='__main__':unittest.main()
