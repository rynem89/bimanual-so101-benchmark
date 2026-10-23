# Optional SO-101 hex-nut camera mount reference

Source: https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module
Downloaded 2026-10-23; upstream Apache-2.0 license included.
STL SHA-256: `e17a626158951ac8cdf8d960962c1a3c8bf23b64635a02f88b62016fe895cef8`.

The 3,476-triangle mesh bounds are approximately (-64.093,-46.093,0) to
(2,-3.5,35) in STL design units, interpreted as mm from the 32 mm module context.
The upstream assembly instructions attach it to hex-nut recesses in the wrist-roll
housing. It is an adapter mesh, not a camera sensor. The mesh is supplied for
visual/mechanical review only: it is NOT instantiated in the robot scene, has no
mass/collision contribution, and is not registered to the robot link frame.
The initial sensor extrinsic in policy_cameras.py is a simulation estimate, not
a CAD-derived or real-hardware measured calibration. Mechanical registration and
cable/clearance assessment remain prerequisites for a physical installation.
