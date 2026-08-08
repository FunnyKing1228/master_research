"""V16 training launcher — run: py -u run_v16.py"""
import os, sys, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'core'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'safe_fl_microgrid'))

import numpy as np
import torch
from train_sac_microgrid import (
    load_config, create_environment, create_agent,
    train_sac_with_microgrid, plot_microgrid_training_results,
    collect_compute_resources, format_compute_resources,
)
from experiment_manager import create_experiment_from_config

CFG_PATH = os.path.join(ROOT, 'configs', 'experiments', 'p302', 'config_p302_v16.yaml')

print("=" * 60)
print("  P302 SLFB — V16 Training (Flow-Scaled Profit Scenario)")
print("  Battery: 2Ah/11.2Wh, 1A, Load: 0.1W x4, Solar: measured")
print("  Flow   : standby=0%, active>=60%, pump power=1/10 measured")
print("=" * 60)

cfg = load_config(CFG_PATH)
print(f"  Config  : {CFG_PATH}")
print(f"  Reward  : {cfg['env']['reward_version']}")
print(f"  Episodes: {cfg['training']['total_episodes']}")
print(f"  Battery : {cfg['env']['battery_capacity_kwh']*1000:.1f} Wh, "
      f"{cfg['env']['battery_power_kw']*1000:.1f} W")
print(f"  Current : {cfg['env']['flow_I_rated_A']} A")
print(f"  Flow    : enabled={cfg['env'].get('use_flow_rate_action')}, "
      f"min_active={cfg['env'].get('flow_min_active_fraction', 0.0)*100:.0f}%, "
      f"Ppump100={cfg['env'].get('flow_P_max_pump_W', 0.0):.3f} W")
print(f"  Dataset : {cfg['env']['dataset_csv_path']}")

exp_manager = create_experiment_from_config(CFG_PATH, experiment_name="sac_v16_profit")

torch.manual_seed(cfg['random_seed'])
np.random.seed(cfg['random_seed'])

env = create_environment(cfg)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

device_cfg = cfg.get('device', 'auto')
device = 'cuda' if (device_cfg == 'auto' and torch.cuda.is_available()) else 'cpu'
print(f"  Device  : {device}")
print(f"  State   : {state_dim}D,  Action: {action_dim}D")
print("=" * 60 + "\n")

agent = create_agent(cfg, state_dim, action_dim, device)

t0 = time.time()
metrics = train_sac_with_microgrid(env, agent, cfg, exp_manager)
elapsed = time.time() - t0

print(f"\n{'='*60}")
print(f"  Training complete in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
if metrics.get('eval_rewards'):
    print(f"  Final eval reward : {metrics['eval_rewards'][-1]:.4f}")
    print(f"  Best eval reward  : {max(metrics['eval_rewards']):.4f}")
print(f"{'='*60}")

compute_res = collect_compute_resources(agent, device, elapsed)
exp_manager.save_results(metrics, metadata={
    'variant': cfg['training'].get('variant', 'sac'),
    'seed': cfg.get('random_seed', 42),
}, compute_resources=compute_res)

print("\n" + format_compute_resources(compute_res))

results_dir = exp_manager.results_dir
plot_microgrid_training_results(metrics, cfg, exp_manager,
                                 save_path=os.path.join(results_dir, "training_results.png"))
print(f"\nResults saved to: {exp_manager.experiment_dir}")
