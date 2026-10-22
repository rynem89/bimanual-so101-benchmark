"""Local LeRobot v0.6.1 conversion, official-policy training and inference.

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
        meta=json.loads(path.read_text());attempts.append(dict(path=str(path.parent),success=meta['success'],case=meta['case']))
        if meta['success']:
            validate_episode(path.parent);accepted.append((path.parent,meta))
    if not accepted:raise RuntimeError('No physically successful demonstrations')
    splits={meta.get('split','diagnostic_legacy') for _,meta in accepted}
    if len(splits)!=1:raise ValueError('Do not mix training/validation/evaluation partitions')
    cameras=list(accepted[0][1]['cameras'])
    features={'observation.state':dict(dtype='float32',shape=(24,),names=JOINT_ORDER+[n+'.velocity' for n in JOINT_ORDER]),
              'action':dict(dtype='float32',shape=(12,),names=JOINT_ORDER)}
    features.update({f'observation.images.{name}':dict(dtype='image',shape=(224,224,3),names=['height','width','channels'])
                     for name in cameras})
    dataset=LeRobotDataset.create('local/so101',fps=20,features=features,root=args.output,
        robot_type='bimanual_so101',use_videos=False,image_writer_threads=4,video_backend='pyav')
    provenance=[]
    try:
        for root,meta in accepted:
            if list(meta['cameras'])!=cameras:raise ValueError('Inconsistent camera configuration')
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
    (args.output/'conversion.json').write_text(json.dumps(manifest,indent=2))
    # Reader verification catches unflushed parquet/schema errors independently.
    loaded=LeRobotDataset('local/so101',root=args.output,video_backend='pyav')
    assert len(loaded)==dataset.num_frames and loaded[0]['action'].shape==(12,)
    assert loaded[0]['observation.state'].shape==(24,)
    for name in cameras:assert loaded[0][f'observation.images.{name}'].shape==(3,224,224)
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


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.configs.types import FeatureType,PolicyFeature
    if args.output.exists():raise FileExistsError('Choose a new checkpoint directory')
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    base=LeRobotDataset('local/so101',root=args.dataset,video_backend='pyav')
    cameras=[key for key in base.features if key.startswith('observation.images.')]
    inputs={'observation.state':PolicyFeature(FeatureType.STATE,(24,))}
    inputs.update({key:PolicyFeature(FeatureType.VISUAL,(3,224,224)) for key in cameras})
    Config,Policy,processors=policy_type(args.method)
    common=dict(device='cuda',input_features=inputs,output_features={'action':PolicyFeature(FeatureType.ACTION,(12,))},
                pretrained_backbone_weights=None if args.no_pretrained else 'ResNet18_Weights.IMAGENET1K_V1')
    # Smaller documented official architectures for the first measured baseline.
    if args.method=='act':
        cfg=Config(**common,chunk_size=40,n_action_steps=20,dim_model=256,n_heads=8,
                   dim_feedforward=1024,n_encoder_layers=4,n_decoder_layers=1)
    else:
        cfg=Config(**common,horizon=32,n_action_steps=16,down_dims=(128,256,512),
                   num_inference_steps=10,noise_scheduler_type='DDIM',drop_n_last_frames=15)
    delta={'action':[i/20 for i in cfg.action_delta_indices]}
    if cfg.observation_delta_indices is not None:
        for key in inputs:delta[key]=[i/20 for i in cfg.observation_delta_indices]
    dataset=LeRobotDataset('local/so101',root=args.dataset,delta_timestamps=delta,video_backend='pyav')
    policy=Policy(cfg).to('cuda');pre,post=processors(cfg,dataset.meta.stats)
    optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=args.lr,weight_decay=1e-4)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                      pin_memory=True,drop_last=True)
    if len(loader)==0:raise RuntimeError('Dataset is smaller than one minibatch')
    iterator=iter(loader);started=time.perf_counter();history=[]
    policy.train()
    for step in range(1,args.steps+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        loss,details=policy(pre(batch))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(policy.parameters(),10.)
        if not torch.isfinite(gradient):raise RuntimeError('Nonfinite training gradient')
        optimizer.step()
        if step==1 or step%args.log_every==0 or step==args.steps:
            row=dict(step=step,loss=loss.item(),gradient_norm=float(gradient),wall_s=time.perf_counter()-started)
            history.append(row);print(json.dumps(row),flush=True)
            (args.output/'progress.json').write_text(json.dumps(history,indent=2))
        if step%500==0:
            saved=args.output/f'step_{step:06d}'
            policy.save_pretrained(saved);pre.save_pretrained(saved);post.save_pretrained(saved)
    checkpoint=args.output/'pretrained_model';policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint);post.save_pretrained(checkpoint)
    # Retain optimizer/Torch RNG for diagnosis and future resume support.
    torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),step=args.steps),args.output/'training_state.pt')
    policy.eval();policy.reset()
    observation={key:base[0][key] for key in inputs}
    with torch.inference_mode():action=post(policy.select_action(pre(observation))).squeeze(0)
    if action.shape!=(12,) or not torch.isfinite(action).all():raise RuntimeError('Invalid saved-policy action')
    report=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='function'},
        method=args.method,official_implementation=Policy.__module__,parameters=sum(p.numel() for p in policy.parameters()),
        dataset_frames=len(dataset),dataset_episodes=dataset.num_episodes,history=history,
        total_wall_s=time.perf_counter()-started,device=torch.cuda.get_device_name(),torch=torch.__version__,
        source_sha256=SCRIPT_SHA256,
        smoke_action_rad=action.cpu().tolist(),note='Loss and finite actions are not manipulation success')
    (args.output/'training.json').write_text(json.dumps(report,indent=2))
    print('[CHECKPOINT]',checkpoint,flush=True)


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
                        observation.update({f'observation.images.{key}':torch.from_numpy(np.frombuffer(value,dtype=np.uint8).reshape(224,224,3).copy()).permute(2,0,1).float()/255
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
                                    Image.fromarray(np.frombuffer(pixels,dtype=np.uint8).reshape(224,224,3)).save(trace_root/camera/f'{trace_index:06d}.png')
                            trace_index+=1
                        connection.send(dict(action=action.tolist(),inference_wall_s=model_s))
    finally:
        if trace:trace.close()
        if args.socket.exists():args.socket.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('convert');p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.set_defaults(function=convert)
    p=commands.add_parser('train');p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=('act','diffusion'),default='act');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=701);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--log-every',type=int,default=100);p.add_argument('--no-pretrained',action='store_true')
    p.set_defaults(function=train)
    p=commands.add_parser('serve');p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--method',choices=('act','diffusion'),default='act');p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--trace-dir',type=Path,help='Optional ignored output directory for pre-action proprioception/targets')
    p.add_argument('--trace-images',action='store_true',help='Also save the RGB policy inputs for diagnostic video')
    p.set_defaults(function=serve)
    args=parser.parse_args()
    if getattr(args,'trace_images',False) and not args.trace_dir:parser.error('--trace-images requires --trace-dir')
    args.function(args)


if __name__=='__main__':main()
