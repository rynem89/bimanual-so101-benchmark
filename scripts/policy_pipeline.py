"""Local LeRobot v0.6.1 conversion, official-training command preparation and inference.

Run only in so101-il-py312. The simulator stays in leisaac-sim and exchanges
RGB/proprioception and 12 absolute joint targets over a local Unix socket.
Nothing is uploaded to the Hub. See --help for each subcommand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np

# Keep model/dataset caches local and ignored, including in a restricted shell.
os.environ.setdefault('HF_HOME',str(Path.cwd()/'outputs/huggingface'))
os.environ.setdefault('TORCH_HOME',str(Path.cwd()/'outputs/torch'))
SCRIPT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

from dataset_io import JOINT_ORDER, TASK, validate_episode


def convert(args):
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    attempts=[];accepted=[]
    for root in sorted(args.raw.iterdir()):
        if root.is_dir() and not (root/'episode.json').exists():
            attempts.append(dict(path=str(root),success=False,status='incomplete'))
    for path in sorted(args.raw.glob('*/episode.json')):
        meta=json.loads(path.read_text());attempts.append(dict(path=str(path.parent),success=meta['success'],case=meta['case'],
            failure=meta.get('evaluation',{}).get('failure'),source_revision=meta.get('source_revision')))
        if meta['success']:
            validate_episode(path.parent);accepted.append((path.parent,meta))
    if not accepted:raise RuntimeError('No physically successful demonstrations')
    splits={meta.get('split','diagnostic_legacy') for _,meta in accepted}
    if len(splits)!=1:raise ValueError('Do not mix training/validation/evaluation partitions')
    cameras=list(accepted[0][1]['cameras'])
    height,width,channels=accepted[0][1]['image_shape']
    features={'observation.state':dict(dtype='float32',shape=(24,),names=JOINT_ORDER+[n+'.velocity' for n in JOINT_ORDER]),
              'action':dict(dtype='float32',shape=(12,),names=JOINT_ORDER)}
    features.update({f'observation.images.{name}':dict(dtype='image',shape=(height,width,channels),names=['height','width','channels'])
                     for name in cameras})
    dataset=LeRobotDataset.create('local/so101',fps=20,features=features,root=args.output,
        robot_type='bimanual_so101',use_videos=False,image_writer_threads=4,video_backend='pyav')
    provenance=[]
    try:
        for root,meta in accepted:
            if list(meta['cameras'])!=cameras or meta['image_shape']!=[height,width,channels]:
                raise ValueError('Inconsistent camera configuration')
            with np.load(root/'trajectory.npz') as data:
                for i,action in enumerate(data['action']):
                    frame={'observation.state':data['state'][i].copy(),'action':action.copy(),'task':TASK}
                    frame.update({f'observation.images.{name}':np.asarray(Image.open(root/name/f'{i:06d}.png'))
                                  for name in cameras})
                    dataset.add_frame(frame)
                dataset.save_episode()
            provenance.append(dict(raw=str(root),case=meta['case'],configuration=meta['configuration'],
                source_revision=meta['source_revision'],source_sha256=meta['source_sha256'],
                raw_schema=meta['schema'],tracked_source_dirty=meta.get('tracked_source_dirty'),
                raw_sha256=hashlib.sha256((root/'trajectory.npz').read_bytes()).hexdigest()))
    finally:dataset.finalize()
    manifest=dict(schema='so101-lerobot-conversion-1.0',lerobot='0.6.1',attempts=attempts,
        accepted_episodes=provenance,cameras=accepted[0][1]['cameras'],privileged_in_training=False,
        frames=dataset.num_frames,episodes=dataset.num_episodes)
    manifest['split']=next(iter(splits))
    manifest['quality_report']=dict(attempted=len(attempts),accepted=len(accepted),
        completed_failures=sum(not row['success'] and row.get('status')!='incomplete' for row in attempts),
        incomplete=sum(row.get('status')=='incomplete' for row in attempts),
        raw_contract_validated=True,visual_quality='Requires human review; nonblank RGB is not visibility evidence')
    (args.output/'conversion.json').write_text(json.dumps(manifest,indent=2))
    # Reader verification catches unflushed parquet/schema errors independently.
    loaded=LeRobotDataset('local/so101',root=args.output,video_backend='pyav')
    assert len(loaded)==dataset.num_frames and loaded[0]['action'].shape==(12,)
    assert loaded[0]['observation.state'].shape==(24,)
    for name in cameras:assert loaded[0][f'observation.images.{name}'].shape==(channels,height,width)
    print(json.dumps(dict(episodes=loaded.num_episodes,frames=len(loaded),schema=loaded.meta.info.codebase_version)))


def policy_type(name):
    if name=='act':
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        return ACTConfig,ACTPolicy,make_act_pre_post_processors
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
    return DiffusionConfig,DiffusionPolicy,make_diffusion_pre_post_processors


def prepare_training(args):
    """Validate feature metadata and print an official CLI command; never train."""
    import shlex
    import sys
    info=json.loads((args.dataset/'meta/info.json').read_text())
    conversion=json.loads((args.dataset/'conversion.json').read_text())
    if conversion['split']!='train':raise ValueError('Training data must have train split provenance')
    expected={'observation.images.overview','observation.images.left_wrist','observation.images.right_wrist'}
    if {k for k in info['features'] if k.startswith('observation.images.')}!=expected:
        raise ValueError('Require three-camera v2 data; human quality approval is a separate gate')
    if args.steps<1 or args.batch_size<1 or not 0<=args.validation_fraction<1:
        raise ValueError('Invalid user-selected training budget/split')
    if args.eval_steps<0 or (args.eval_steps>0 and args.validation_fraction==0):
        raise ValueError('Evaluation cadence requires a nonzero validation split')
    command=[sys.executable,'-m','lerobot.scripts.lerobot_train',
        '--dataset.repo_id=local/so101','--dataset.root='+str(args.dataset),
        '--policy.type='+args.method,'--policy.push_to_hub=false',
        '--output_dir='+str(args.output),'--steps='+str(args.steps),
        '--batch_size='+str(args.batch_size),'--dataset.eval_split='+str(args.validation_fraction),
        '--eval_steps='+str(args.eval_steps),'--env_eval_freq=0','--wandb.enable=false']
    print('Review only. Execute separately after camera/data approval and training authorization:')
    print(shlex.join(command))


def serve(args):
    import torch
    from multiprocessing.connection import Listener
    from lerobot.policies.factory import make_pre_post_processors
    Config,Policy,_=policy_type(args.method)
    policy=Policy.from_pretrained(args.checkpoint).to('cuda').eval()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=args.checkpoint)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    metadata=dict(checkpoint=str(args.checkpoint.resolve()),method=args.method,
                  weights_sha256=hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                  adapter_sha256=SCRIPT_SHA256,torch=torch.__version__,lerobot='0.6.1',
                  input_trace=bool(args.trace_dir),trace_images=args.trace_images)
    cameras={key.removeprefix('observation.images.') for key in policy.config.input_features
             if key.startswith('observation.images.')}
    shapes={key.removeprefix('observation.images.'):tuple(feature.shape) for key,feature in policy.config.input_features.items() if key.startswith('observation.images.')}
    trace=None;trace_root=None;trace_index=0
    os.umask(0o077)
    if args.socket.exists():raise FileExistsError('Refuse to overwrite an existing socket')
    try:
        with Listener(str(args.socket),family='AF_UNIX') as listener:
            print('[READY]',args.socket,flush=True)
            while True:
                with listener.accept() as connection:
                    while True:
                        try:request=connection.recv()
                        except EOFError:break
                        if request['command']=='reset':
                            seed=int(request.get('seed',701));torch.manual_seed(seed);np.random.seed(seed%2**32);random.seed(seed)
                            policy.reset()
                            if trace:trace.close()
                            trace=None;trace_index=0
                            if args.trace_dir:
                                trace_root=args.trace_dir/f'seed_{seed}_{time.time_ns()}'
                                trace_root.mkdir(parents=True,exist_ok=False)
                                (trace_root/'input_trace.json').write_text(json.dumps(dict(schema='so101-policy-input-trace-1.0',
                                    seed=seed,policy_metadata=metadata,joint_order=JOINT_ORDER,
                                    note='Pre-action inputs and requested targets only; no applied-target proof, terminal image or evaluator labels. Not demonstration data.'),indent=2))
                                trace=(trace_root/'actions.jsonl').open('w')
                                if args.trace_images:
                                    for camera in cameras:(trace_root/camera).mkdir()
                            connection.send(dict(ok=True,protocol='so101-policy-1.0',seed=seed,policy_metadata=metadata));continue
                        if request['command']=='close':connection.send({'ok':True});return
                        if request['command']!='action':raise ValueError('Unknown inference command')
                        started=time.perf_counter()
                        # Allowlist prevents accidental object/contact/phase labels.
                        if set(request['images'])!=cameras or np.shape(request['state'])!=(24,) or not np.isfinite(request['state']).all():
                            raise ValueError('Expected calibrated RGB cameras and finite 24-value proprioception')
                        observation={'observation.state':torch.tensor(request['state'],dtype=torch.float32)}
                        observation.update({f'observation.images.{key}':torch.from_numpy(np.frombuffer(value,dtype=np.uint8).reshape(*shapes[key][1:],3).copy()).permute(2,0,1).float()/255
                                            for key,value in request['images'].items()})
                        with torch.inference_mode():action=post(policy.select_action(pre(observation))).squeeze(0).cpu().numpy()
                        if action.shape!=(12,) or not np.isfinite(action).all():raise RuntimeError('Invalid policy action')
                        model_s=time.perf_counter()-started
                        if trace:
                            trace.write(json.dumps(dict(policy_step=trace_index,state=request['state'],
                                target_rad=action.tolist(),model_s=model_s),allow_nan=False)+'\n');trace.flush()
                            if args.trace_images:
                                from PIL import Image
                                for camera,pixels in request['images'].items():
                                    Image.fromarray(np.frombuffer(pixels,dtype=np.uint8).reshape(*shapes[camera][1:],3)).save(trace_root/camera/f'{trace_index:06d}.png')
                            trace_index+=1
                        connection.send(dict(action=action.tolist(),inference_wall_s=model_s))
    finally:
        if trace:trace.close()
        if args.socket.exists():args.socket.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('convert');p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.set_defaults(function=convert)
    p=commands.add_parser('prepare-training',help='Print official training command only; never execute updates')
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=('act','diffusion'),required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--validation-fraction',type=float,required=True);p.add_argument('--eval-steps',type=int,required=True)
    p.set_defaults(function=prepare_training)
    p=commands.add_parser('serve');p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--method',choices=('act','diffusion'),default='act');p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--trace-dir',type=Path,help='Optional ignored output directory for pre-action proprioception/targets')
    p.add_argument('--trace-images',action='store_true',help='Also save the RGB policy inputs for diagnostic video')
    p.set_defaults(function=serve)
    args=parser.parse_args()
    if getattr(args,'trace_images',False) and not args.trace_dir:parser.error('--trace-images requires --trace-dir')
    args.function(args)


if __name__=='__main__':main()
