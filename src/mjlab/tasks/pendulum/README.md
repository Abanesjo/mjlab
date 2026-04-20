## Training
```
python src/mjlab/scripts/train.py Mjlab-Pendulum-Balance-Unitree-Go2 --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.run_name curriculum_v3 --video True
```

## Playing
```
python src/mjlab/scripts/play.py Mjlab-Pendulum-Balance-Unitree-Go2 --checkpoint-file logs/rsl_rl/go2_pendulum/frame_stacking/model_20000.pt --num_envs 1 --viewer native
```