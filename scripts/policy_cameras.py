"""Project-owned camera sensors; no robot asset, collision or mass modifications.

Initial simulation extrinsics, not measured hardware calibration. Optical axes
use ROS: +X right, +Y down, +Z forward. Distances are metres.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw

NAMES = ('overview', 'left_wrist', 'right_wrist')
CALIBRATION_PATH = Path(__file__).resolve().parents[1]/'configs/policy_cameras.json'
CALIBRATION_BYTES = CALIBRATION_PATH.read_bytes()
CALIBRATION = json.loads(CALIBRATION_BYTES)
CALIBRATION_SHA256 = hashlib.sha256(CALIBRATION_BYTES).hexdigest()
WIDTH, HEIGHT = CALIBRATION['resolution_wh']
APPROVED_PROFILE = CALIBRATION['overview_profile']
# Candidate poses are fixed in world metres for an entire run. Wrist geometry
# and intrinsics stay identical across this overview-only comparison.
OVERVIEW_PROFILES = {
    'baseline': ((0., -.10, .72), (0., -.10, .02)),
    'centered': (tuple(CALIBRATION['overview_eye_world_m']), tuple(CALIBRATION['overview_target_world_m'])),
    'angled_front': ((0., .07, .78), (0., -.18, .02)),
    'angled_rear': ((0., -.45, .78), (0., -.18, .02)),
    'angled_left': ((-.25, -.18, .78), (0., -.18, .02)),
}
# Wrist roll housing is the rigid gripper link, not the articulated moving jaw.
WRIST_POSITION = np.array(CALIBRATION['wrist_position_parent_m'])
WRIST_TARGET = np.array(CALIBRATION['wrist_target_parent_m'])


def optical_rotation(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Choose optical right along negative link X and point optical Z at the grasp region."""
    z = target-position; z = z/np.linalg.norm(z)
    x = np.cross(np.array([0., 1., 0.]), z); x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


WRIST_ROTATION = optical_rotation(WRIST_POSITION, WRIST_TARGET)
WRIST_QUAT = tuple(CALIBRATION['wrist_quaternion_ros_wxyz'])


def quaternion_rotation(quaternion) -> np.ndarray:
    """Unit wxyz quaternion to rotation, without a simulation-only SciPy dependency."""
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-5:
        raise ValueError('Invalid unit optical quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


if not np.allclose(quaternion_rotation(WRIST_QUAT),WRIST_ROTATION,atol=1e-12,rtol=0):
    raise ValueError('Frozen wrist optical target and orientation disagree')


def configure_cameras(scene_cfg, diagnostic_front: bool=False, compare_overviews: bool=False):
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
            spawn=PinholeCameraCfg(focal_length=CALIBRATION['focal_length_mm'],
                                  horizontal_aperture=CALIBRATION['horizontal_aperture_mm'],
                                  clipping_range=tuple(CALIBRATION['clipping_range_m']))))

    if compare_overviews:
        # Diagnostic-only sensors never enter NAMES or the policy observation.
        for profile in OVERVIEW_PROFILES:
            setattr(scene_cfg, 'review_'+profile, CameraCfg(
                class_type=ArticulatedCamera, prim_path='{ENV_REGEX_NS}/review_'+profile,
                update_period=0., height=HEIGHT,width=WIDTH,data_types=['rgb'],
                update_latest_camera_pose=True,
                spawn=PinholeCameraCfg(focal_length=12.,horizontal_aperture=20.955,
                                      clipping_range=(.01,10.))))

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


class CaptureContinuity:
    """Check every policy frame, including motion between sparse phase images."""
    def __init__(self):
        self.last_step = None
        self.last_frame_ids = None
        self.count = 0
        self.failure = None

    def _fail(self, message: str):
        self.failure=message
        raise ValueError(message)

    def accept(self, step: int, origin: int, frame_ids: dict) -> bool:
        if step < origin or (step-origin)%3:
            self._fail('Camera timestamp is not a 20 Hz boundary')
        if set(frame_ids) != set(NAMES):
            self._fail('Missing policy frame identifiers')
        if step == self.last_step:
            return False  # Initial phase and first action share a physical tick.
        if self.last_step is not None:
            if step-self.last_step != 3:
                self._fail('Missing or reordered policy camera frame')
            if any(frame_ids[n] <= self.last_frame_ids[n] for n in NAMES):
                self._fail('Camera sensor frame did not advance')
        self.last_step = step
        self.last_frame_ids = frame_ids.copy()
        self.count += 1
        return True


class CameraAudit:
    """Actual tensor previews; comparison adds continuous camera diagnostics only."""
    def __init__(self, root: Path, compare_overviews: bool=False):
        self.root=root; root.mkdir(parents=True,exist_ok=False)
        self.seen=set();self.rows=[];self.last_phase=None;self.phase_ticks=0
        self.compare_overviews=compare_overviews
        self.continuity=CaptureContinuity()
        self.video=None;self.continuity_file=None
        self.hashes={name:set() for name in NAMES}
        if compare_overviews:
            import imageio_ffmpeg
            self.video=imageio_ffmpeg.write_frames(str(root/'policy_streams_20hz.mp4'),
                (WIDTH*3,HEIGHT),fps=20,codec='libx264',quality=6,
                output_params=['-preset','veryfast','-threads','2'])
            self.video.send(None)
            self.continuity_file=(root/'continuity.jsonl').open('w')

    def capture(self, env, phase: str, force: bool=False):
        if phase != self.last_phase:self.phase_ticks=0;self.last_phase=phase
        self.phase_ticks+=3
        key=(phase, 'entry' if self.phase_ticks==3 else 'settled')
        sparse=force or (key not in self.seen and (key[1]=='entry' or self.phase_ticks>=60))
        if not sparse and not self.compare_overviews:return
        if self.compare_overviews and env.total_steps==self.continuity.last_step:return
        images=env.capture_images();check_capture(images,env.total_steps,env.policy_origin_step)
        metadata=env.camera_metadata()  # Independently checks both wrist attachments.
        if self.compare_overviews:
            import hashlib
            ids={name:int(env.cameras[name].frame[0]) for name in NAMES}
            self.continuity.accept(env.total_steps,env.policy_origin_step,ids)
            digests={name:hashlib.sha256(rgb.tobytes()).hexdigest() for name,rgb in images.items()}
            for name,digest in digests.items():self.hashes[name].add(digest)
            self.continuity_file.write(json.dumps(dict(index=self.continuity.count-1,
                physics_step=env.total_steps,relative_time_s=(env.total_steps-env.policy_origin_step)/60,
                sensor_frame_ids=ids,rgb_sha256=digests,cameras=metadata))+'\n')
            self.continuity_file.flush()
            self.video.send(np.concatenate([images[n] for n in NAMES],axis=1))
        if not sparse:return
        row=dict(phase=phase,sample=key[1],physics_step=env.total_steps,
                 time_s=env.total_steps/60,cameras=metadata)
        canvas=Image.new('RGB',(WIDTH*3,HEIGHT+28),'white')
        for i,name in enumerate(NAMES):
            canvas.paste(Image.fromarray(images[name]),(i*WIDTH,28))
            ImageDraw.Draw(canvas).text((i*WIDTH+8,6),f'{phase} {key[1]} | {name} | tick {env.total_steps}',fill='black')
        file=f'{len(self.rows):03d}_{phase}_{key[1]}.jpg'
        canvas.save(self.root/file,quality=95);row['image']=file
        if self.compare_overviews:
            candidates=env.capture_overviews()
            comparison=Image.new('RGB',(WIDTH*len(candidates),HEIGHT+28),'white')
            measures={}
            for i,(name,rgb) in enumerate(candidates.items()):
                comparison.paste(Image.fromarray(rgb),(i*WIDTH,28))
                ImageDraw.Draw(comparison).text((i*WIDTH+8,6),f'{phase} | {name} | tick {env.total_steps}',fill='black')
                measures[name]=overview_measurement(rgb,env.scene.sensors['review_'+name],env.frame)
            name=file.removesuffix('.jpg')+'_overviews.jpg'
            comparison.save(self.root/name,quality=95)
            row.update(overview_comparison=name,overview_measurements=measures)
        if 'front' in env.scene.sensors:
            front=env.scene.sensors['front'];front.update(0.,force_recompute=True)
            rgb=front.data.output['rgb'][0,:,:,:3].cpu().numpy()
            Image.fromarray(rgb).save(self.root/(file.removesuffix('.jpg')+'_diagnostic_front.png'))
        self.rows.append(row);self.seen.add(key)
        (self.root/'audit.json').write_text(json.dumps(dict(schema='so101-camera-audit-2',
            note='Camera diagnostics only; no demonstration targets or training data',
            overview_profiles=OVERVIEW_PROFILES,samples=self.rows),indent=2))

    def finish(self):
        if self.video is not None:self.video.close();self.video=None
        if self.continuity_file is not None:self.continuity_file.close();self.continuity_file=None
        if self.compare_overviews:
            (self.root/'continuity_summary.json').write_text(json.dumps(dict(
                observations=self.continuity.count,physics_ticks_per_observation=3,fps=20,
                missing_or_reordered_frames=0 if self.continuity.failure is None else None,
                validation_failure=self.continuity.failure,
                attachment_checks='passed for every logged camera frame; physical termination is reported separately',
                unique_rgb_frames={name:len(hashes) for name,hashes in self.hashes.items()},
                note='Physical step stamps are software synchronization checks, not independent RTX exposure timestamps.'),indent=2))
            build_overview_review(self.root)


def overview_measurement(rgb: np.ndarray, camera, frame) -> dict:
    """Diagnostic ground-truth projection plus a red-pixel proxy; never policy input."""
    from itertools import product
    pose=frame.cube_pose()
    corners=np.array(list(product((-.02,.02),repeat=3)))@pose[:3,:3].T+pose[:3,3]
    q=camera.data.quat_w_ros[0].cpu().numpy()
    rotation=quaternion_rotation(q)
    local=(corners-camera.data.pos_w[0].cpu().numpy())@rotation
    k=camera.data.intrinsic_matrices[0].cpu().numpy()
    pixels=local@k.T;pixels=pixels[:,:2]/pixels[:,2,None]
    inside=(local[:,2]>.01)&(pixels[:,0]>=0)&(pixels[:,0]<WIDTH)&(pixels[:,1]>=0)&(pixels[:,1]<HEIGHT)
    # Only the cube is red in this nominal scene. This is not a segmentation
    # ground truth or certified visibility fraction under appearance changes.
    red=(rgb[:,:,0]>100)&(rgb[:,:,0]>1.35*rgb[:,:,1])&(rgb[:,:,0]>1.35*rgb[:,:,2])
    return dict(cube_all_projected_corners_in_frame=bool(inside.all()),
        cube_projected_bounds_px=[pixels.min(0).tolist(),pixels.max(0).tolist()],
        red_pixel_proxy=int(red.sum()),
        position_world_m=camera.data.pos_w[0].cpu().tolist(),
        quaternion_ros_wxyz=q.tolist(),intrinsic=k.tolist())


def build_overview_review(root: Path):
    audit=json.loads((root/'audit.json').read_text())
    phases=['initial','pregrasp','close','hold','right_close','right_only_hold','right_to_bin','bin_release','bin_settle']
    selected=[]
    for phase in phases:
        rows=[r for r in audit['samples'] if r['phase']==phase]
        if rows:selected.append(rows[-1])
    # One artifact compares ALL alternatives at identical physical times.
    from PIL import ImageFont
    font=ImageFont.truetype('DejaVuSans.ttf',17)
    small=ImageFont.truetype('DejaVuSans.ttf',14)
    sheet=Image.new('RGB',(320*len(OVERVIEW_PROFILES),70+270*len(selected)),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    for j,(name,(eye,target)) in enumerate(OVERVIEW_PROFILES.items()):
        draw.text((j*320+8,5),name,font=font,fill='black')
        draw.text((j*320+8,29),f'eye {eye} m',font=small,fill='black')
        draw.text((j*320+8,47),f'target {target} m',font=small,fill='black')
    for i,row in enumerate(selected):
        original=Image.open(root/row['overview_comparison'])
        y=70+270*i
        draw.text((8,y+4),f"{row['phase']} | physics tick {row['physics_step']} (all views identical time)",font=font,fill='black')
        for j in range(len(OVERVIEW_PROFILES)):
            cell=original.crop((j*WIDTH,28,(j+1)*WIDTH,HEIGHT+28)).resize((320,240),Image.Resampling.LANCZOS)
            sheet.paste(cell,(j*320,y+30))
    sheet.save(root/'overview_comparison.jpg',quality=95)
    scores={name:dict(full_cube_in_frame_samples=sum(r['overview_measurements'][name]['cube_all_projected_corners_in_frame'] for r in audit['samples']),
        samples=len(audit['samples']),red_pixel_sum=sum(r['overview_measurements'][name]['red_pixel_proxy'] for r in selected)) for name in OVERVIEW_PROFILES}
    (root/'comparison_summary.json').write_text(json.dumps(dict(profiles=OVERVIEW_PROFILES,
        selected_ticks={r['phase']:r['physics_step'] for r in selected},measurements=scores,
        warning='Projection tests frustum bounds only. Red pixels are a colour proxy; human occlusion/framing review is required.'),indent=2))


def build_recommended_review(root: Path, profile: str):
    """Consolidated same-tick original/proposal/wrist sheet; no simulator runs."""
    import hashlib
    from PIL import ImageFont
    audit=json.loads((root/'audit.json').read_text())
    profiles=list(audit['overview_profiles'])
    column=profiles.index(profile)
    phases=['initial','pregrasp','close','hold','right_close','right_only_hold','right_to_bin','bin_release','bin_settle']
    selected=[[r for r in audit['samples'] if r['phase']==phase][-1] for phase in phases]
    font=ImageFont.truetype('DejaVuSans.ttf',17)
    small=ImageFont.truetype('DejaVuSans.ttf',14)
    sheet=Image.new('RGB',(1280,90+270*len(selected)),(245,245,245))
    draw=ImageDraw.Draw(sheet)
    headings=['Original overview', 'Proposed '+profile, 'Left wrist (unchanged)', 'Right wrist (unchanged)']
    for j,label in enumerate(headings):draw.text((j*320+8,5),label,font=font,fill='black')
    eye,target=audit['overview_profiles'][profile]
    draw.text((8,33),f'Proposed world eye {eye} m; target {target} m | 640x480 RGB | 20 Hz | final approval pending',font=small,fill='black')
    draw.text((8,54),'All four cells in each row share the same physics tick. Wrist images are the approved original sensor streams.',font=small,fill='black')
    for i,row in enumerate(selected):
        y=90+270*i
        draw.text((8,y+4),f"{row['phase']} | physics tick {row['physics_step']}",font=font,fill='black')
        overview=Image.open(root/row['overview_comparison'])
        policy=Image.open(root/row['image'])
        cells=[overview.crop((0,28,WIDTH,HEIGHT+28)),
               overview.crop((column*WIDTH,28,(column+1)*WIDTH,HEIGHT+28)),
               policy.crop((WIDTH,28,2*WIDTH,HEIGHT+28)),
               policy.crop((2*WIDTH,28,3*WIDTH,HEIGHT+28))]
        for j,cell in enumerate(cells):sheet.paste(cell.resize((320,240),Image.Resampling.LANCZOS),(j*320,y+30))
    sheet.save(root/'recommended_review.jpg',quality=95)
    measured=selected[0]['overview_measurements'][profile]
    calibration=dict(status='proposed; awaiting final human approval',overview_profile=profile,
        overview_eye_world_m=eye,overview_target_world_m=target,
        overview_quaternion_ros_wxyz=measured['quaternion_ros_wxyz'],
        resolution=[WIDTH,HEIGHT],fps=20,focal_length_mm=12.,horizontal_aperture_mm=20.955,
        clipping_range_m=[.01,10.],intrinsic=measured['intrinsic'],
        wrist_position_gripper_m=WRIST_POSITION.tolist(),wrist_quaternion_ros_wxyz=list(WRIST_QUAT),
        selected_ticks={r['phase']:r['physics_step'] for r in selected})
    frozen_fields={k:v for k,v in calibration.items() if k not in ('status','selected_ticks')}
    calibration['calibration_sha256']=hashlib.sha256(json.dumps(frozen_fields,sort_keys=True).encode()).hexdigest()
    (root/'proposed_calibration.json').write_text(json.dumps(calibration,indent=2))


def inspect_scene(layout, args):
    """Stationary viewer using the exact task assets and camera capture interface.

    Reset performs the established three-second settling hold. Thereafter only
    rendering advances: no expert, action sequence, recording or task execution.
    Optical guides are display-only and hidden while the actual RGB is captured.
    """
    import gc
    import time
    from pxr import Gf, UsdGeom
    import omni.usd
    from benchmark_env import BenchmarkEnv
    from so101_model import SO101Model, calibrate

    if args.snapshot_dir and args.snapshot_dir.exists():
        raise FileExistsError(args.snapshot_dir)
    model = SO101Model(layout.ROBOT_USD)
    calibrate(model)  # Mesh-derived TCP only; never solves or executes an IK path.
    sim = layout.SimulationContext(layout.sim_utils.SimulationCfg(device=layout.args.device))
    cfg = layout.BimanualSceneCfg(num_envs=1, env_spacing=2.)
    configure_cameras(cfg)
    cfg.left_arm.init_state.pos = layout.LEFT_BASE_M
    cfg.right_arm.init_state.pos = layout.RIGHT_BASE_M
    cfg.left_arm.spawn.activate_contact_sensors = True
    cfg.right_arm.spawn.activate_contact_sensors = True
    cfg.cube.spawn.activate_contact_sensors = True
    for name in ('table', 'bin_floor', 'bin_left_wall', 'bin_right_wall', 'bin_front_wall', 'bin_back_wall'):
        getattr(cfg, name).spawn.rigid_props = layout.sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, disable_gravity=True)
    scene = layout.InteractiveScene(cfg)
    sim.reset()
    sim.set_camera_view(eye=(.8, .9, .7), target=(0., -.12, .10))
    env = BenchmarkEnv(sim, scene, model, layout)
    window = None
    try:
        env.reset()
        images = env.capture_images()
        metadata = env.camera_metadata()
        from demonstration_pipeline import check_camera
        for name, camera in metadata.items():
            check_camera(camera, name)
        initial_step = env.total_steps
        initial_time = sim.current_time

        def physical_state():
            # Read PhysX directly: cached ArticulationData would not establish
            # that render-only inspection left the real robot/object unchanged.
            joints = np.stack([arm.root_physx_view.get_dof_positions()[0].cpu().numpy().copy()
                               for arm in env.arms])
            cube = env.cube.root_physx_view.get_transforms()[0].cpu().numpy().copy()
            return joints, cube

        initial_joints, initial_cube = physical_state()
        stage = omni.usd.get_context().get_stage()
        guides = UsdGeom.Xform.Define(stage, '/World/InspectionGuides')

        def line(name, points, colour):
            curve = UsdGeom.BasisCurves.Define(stage, '/World/InspectionGuides/' + name)
            curve.CreateTypeAttr('linear')
            curve.CreateCurveVertexCountsAttr([2])
            curve.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
            curve.CreateWidthsAttr([.0015])
            curve.CreateDisplayColorAttr([Gf.Vec3f(*colour)])

        for name, camera in metadata.items():
            position = np.array(camera['position_world_m'])
            rotation = quaternion_rotation(camera['quaternion_ros_wxyz'])
            line(name + '_optical_Z', [position, position + .10 * rotation[:, 2]], (1., .8, .1))
            if args.frames:
                for axis, colour in enumerate(((1., 0., 0.), (0., 1., 0.), (0., .4, 1.))):
                    line(name + '_axis_' + str(axis), [position, position + .035 * rotation[:, axis]], colour)
        if args.frames:
            frames = {'world': np.eye(4), 'left_base': model.base.copy(), 'right_base': model.base.copy(),
                      'left_TCP': env.frame.tcp[0], 'right_TCP': env.frame.tcp[1]}
            frames['right_base'][0, 3] = .25
            for name, pose in frames.items():
                for axis, colour in enumerate(((1., 0., 0.), (0., 1., 0.), (0., .4, 1.))):
                    line(name + str(axis), [pose[:3, 3], pose[:3, 3] + .06 * pose[:3, axis]], colour)

        providers = {}
        if not layout.args.headless:
            import omni.ui as ui
            window = ui.Window('SO-101 actual policy camera feeds', width=1020, height=340)
            groups = {}
            with window.frame:
                with ui.VStack(spacing=6):
                    ui.Label('Stationary inspection: no expert or recording. Optical guides are excluded from RGB.')
                    selector = ui.ComboBox(0, 'All cameras', *NAMES, height=25)
                    with ui.HStack(spacing=8):
                        for name in NAMES:
                            with ui.VStack() as group:
                                groups[name] = group
                                ui.Label(name, height=22)
                                providers[name] = ui.ByteImageProvider()
                                ui.ImageWithProvider(providers[name], width=320, height=240)

            def select_camera(item_model, item):
                choice = item_model.get_item_value_model().as_int
                for index, name in enumerate(NAMES, 1):
                    groups[name].visible = choice in (0, index)
            selector.model.add_item_changed_fn(select_camera)
            for choice in range(4):
                selector.model.get_item_value_model().set_value(choice)
                if any(groups[name].visible != (choice in (0, index)) for index, name in enumerate(NAMES, 1)):
                    raise RuntimeError('Camera selector did not update the feed panels')
            selector.model.get_item_value_model().set_value(0)

        if args.snapshot_dir:
            args.snapshot_dir.mkdir(parents=True)
            for name, pixels in images.items():
                Image.fromarray(pixels).save(args.snapshot_dir / (name + '.png'))
        print('[SCENE] Settled and frozen. Three actual RGB feeds enabled; no expert or recording.', flush=True)
        print('[SCENE] Robot: existing LeIsaac USD. Cameras: gripper children. Mount STL is not physical geometry.', flush=True)
        print('[SCENE] Yellow guides: optical +Z; optional axes: X red / Y green / Z blue. Close Isaac to exit.', flush=True)
        started = time.monotonic()
        captures = 0
        while layout.app.is_running() and (not args.duration or time.monotonic() - started < args.duration):
            guides.MakeInvisible()
            images = env.capture_images()
            current_metadata = env.camera_metadata()
            for name, camera in current_metadata.items():
                check_camera(camera, name)
            for name, provider in providers.items():
                rgba = np.dstack((images[name], np.full((HEIGHT, WIDTH, 1), 255, dtype=np.uint8)))
                provider.set_bytes_data(rgba.flatten().data, [WIDTH, HEIGHT])
            guides.MakeVisible()
            sim.render()
            captures += 1
            time.sleep(.05)
        final_joints, final_cube = physical_state()
        joint_change = float(np.abs(final_joints - initial_joints).max())
        cube_change = float(np.linalg.norm(final_cube[:3] - initial_cube[:3]))
        quaternion_change = float(min(np.linalg.norm(final_cube[3:] - initial_cube[3:]),
                                      np.linalg.norm(final_cube[3:] + initial_cube[3:])))
        report = dict(schema='so101-stationary-scene-check-1.0', calibration_sha256=CALIBRATION_SHA256,
                      cameras=metadata, reset_physics_steps=initial_step,
                      viewer_physics_steps=env.total_steps - initial_step, rendered_captures=captures,
                      viewer_sim_time_change_s=sim.current_time-initial_time,
                      maximum_joint_change_rad=joint_change, cube_translation_change_m=cube_change,
                      cube_quaternion_change=quaternion_change,
                      expert_started=False, recording_started=False, frames_shown=args.frames,
                      gui_panel_created=window is not None,
                      gui_selection_checked=window is not None,
                      note='Static attachment/intrinsics checks only; GUI usability still requires human review.')
        if args.snapshot_dir:
            (args.snapshot_dir / 'scene_check.json').write_text(json.dumps(report, indent=2) + '\n')
        if (not np.isfinite([joint_change, cube_change, quaternion_change, report['viewer_sim_time_change_s']]).all()
                or report['viewer_physics_steps'] != 0 or abs(report['viewer_sim_time_change_s']) > 1e-12
                or max(joint_change, cube_change, quaternion_change) > 1e-7):
            raise RuntimeError('Stationary viewer advanced physical time or changed measured robot/cube state')
        print('[SCENE RESULT]', json.dumps(report), flush=True)
    finally:
        if window is not None:
            window.destroy()
        env.cameras.clear()
        scene.sensors.clear()
        gc.collect()
        sim.clear_all_callbacks()
        sim.clear_instance()
        layout.app.close(wait_for_replicator=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Rebuild camera review figures from saved diagnostic RGB; never simulate or collect.')
    parser.add_argument('--build-review',type=Path,required=True)
    parser.add_argument('--recommend',choices=tuple(OVERVIEW_PROFILES))
    args=parser.parse_args()
    build_overview_review(args.build_review)
    if args.recommend:build_recommended_review(args.build_review,args.recommend)
