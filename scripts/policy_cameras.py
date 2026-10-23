"""Project-owned camera sensors; no robot asset, collision or mass modifications.

Initial simulation extrinsics, not measured hardware calibration. Optical axes
use ROS: +X right, +Y down, +Z forward. Distances are metres.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

NAMES = ('overview', 'left_wrist', 'right_wrist')
HEIGHT, WIDTH = 480, 640
# Wrist roll housing is the rigid gripper link, not the articulated moving jaw.
WRIST_POSITION = np.array([0.0141, -0.045, 0.005])
WRIST_TARGET = np.array([0.0141, 0., -0.09])


def optical_rotation(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Choose optical right along negative link X and point optical Z at the grasp region."""
    z = target-position; z = z/np.linalg.norm(z)
    x = np.cross(np.array([0., 1., 0.]), z); x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


WRIST_ROTATION = optical_rotation(WRIST_POSITION, WRIST_TARGET)
WRIST_QUAT = tuple(Rotation.from_matrix(WRIST_ROTATION).as_quat()[[3,0,1,2]])


def configure_cameras(scene_cfg, diagnostic_front: bool=False):
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    class ArticulatedCamera(Camera):
        """Read live Fabric transforms; pinned Camera defaults to stale USD poses."""
        def _update_poses(self, env_ids):
            pos, quat = self._view.get_world_poses(usd=False)
            self._data.pos_w[env_ids] = pos[env_ids]
            self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
                quat[env_ids], origin="opengl", target="world")

    from isaaclab.sim import PinholeCameraCfg
    for name in NAMES:
        parent = '' if name == 'overview' else ('Left_Robot' if name == 'left_wrist' else 'Right_Robot')+'/gripper/'
        offset = CameraCfg.OffsetCfg() if name == 'overview' else CameraCfg.OffsetCfg(
            pos=tuple(WRIST_POSITION), rot=WRIST_QUAT, convention='ros')
        setattr(scene_cfg, name, CameraCfg(
            class_type=ArticulatedCamera, prim_path='{ENV_REGEX_NS}/'+parent+name, offset=offset,
            update_period=0., height=HEIGHT, width=WIDTH, data_types=['rgb'],
            update_latest_camera_pose=True,
            spawn=PinholeCameraCfg(focal_length=12., horizontal_aperture=20.955,
                                  clipping_range=(.01, 10.))))

    if diagnostic_front:
        scene_cfg.front = CameraCfg(prim_path='{ENV_REGEX_NS}/front', update_period=0.,
            height=HEIGHT,width=WIDTH,data_types=['rgb'],
            spawn=PinholeCameraCfg(focal_length=18.,horizontal_aperture=20.955,clipping_range=(.01,10.)))


def check_capture(images, step: int, origin: int) -> None:
    if set(images) != set(NAMES): raise ValueError('Missing policy camera')
    if step < origin or (step-origin)%3: raise ValueError('Camera timestamp is not a 20 Hz boundary')
    for name, rgb in images.items():
        if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8 or rgb.std() < 2:
            raise ValueError('Invalid RGB camera: '+name)


def check_attachment(position: np.ndarray, rotation: np.ndarray,
                     parent_position: np.ndarray, parent_rotation: np.ndarray) -> None:
    """Compare a rendered camera with its independent measured rigid-link pose."""
    expected_position=parent_position+parent_rotation@WRIST_POSITION
    if not np.allclose(position,expected_position,atol=2e-5,rtol=0):
        raise RuntimeError('Camera attachment translation mismatch')
    if not np.allclose(rotation,parent_rotation@WRIST_ROTATION,atol=2e-4,rtol=0):
        raise RuntimeError('Camera attachment rotation mismatch')


class CameraAudit:
    """Sparse phase snapshots from actual tensors, not demonstrations or viewport screenshots."""
    def __init__(self, root: Path):
        self.root=root; root.mkdir(parents=True,exist_ok=False)
        self.seen=set();self.rows=[];self.last_phase=None;self.phase_ticks=0

    def capture(self, env, phase: str, force: bool=False):
        if phase != self.last_phase:self.phase_ticks=0;self.last_phase=phase
        self.phase_ticks+=3
        # Capture phase entry and a later sample, allowing short grasp phases.
        key=(phase, 'entry' if self.phase_ticks==3 else 'settled')
        if not force and (key in self.seen or (key[1]=='settled' and self.phase_ticks<60)):return
        images=env.capture_images();check_capture(images,env.total_steps,env.policy_origin_step)
        metadata=env.camera_metadata()
        row=dict(phase=phase,sample=key[1],physics_step=env.total_steps,
                 time_s=env.total_steps/60,cameras=metadata)
        canvas=Image.new('RGB',(WIDTH*3,HEIGHT+28),'white')
        for i,name in enumerate(NAMES):
            canvas.paste(Image.fromarray(images[name]),(i*WIDTH,28))
            ImageDraw.Draw(canvas).text((i*WIDTH+8,6),f'{phase} {key[1]} | {name} | tick {env.total_steps}',fill='black')
        file=f'{len(self.rows):03d}_{phase}_{key[1]}.jpg'
        canvas.save(self.root/file,quality=95);row['image']=file
        if 'front' in env.scene.sensors:
            front=env.scene.sensors['front'];front.update(0.,force_recompute=True)
            rgb=front.data.output['rgb'][0,:,:,:3].cpu().numpy()
            Image.fromarray(rgb).save(self.root/(file.removesuffix('.jpg')+'_diagnostic_front.png'))
        self.rows.append(row);self.seen.add(key)
        (self.root/'audit.json').write_text(json.dumps(dict(schema='so101-camera-audit-1',
            note='Diagnostic snapshots only; no demonstration actions or training data',samples=self.rows),indent=2))
