"""Regression scenarios for the PPO locomotion overhaul."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch


def reward_args(**overrides):
    args = dict(base_lin_vel=np.zeros(3), cmd_lin_vel=np.array([0.2, 0, 0]),
                base_ang_vel=np.zeros(3), cmd_ang_vel=0.0,
                proj_gravity=np.array([0, 0, -1]), joint_torques=np.zeros(8),
                joint_vel=np.zeros(8), foot_contact=np.array([1, 1, 0, 0]))
    args.update(overrides)
    return args


def test_reward_dwell_and_weighted_progress_scalar_batch():
    from training.env_config import compute_rewards, LocomotionMode

    rewards = [compute_rewards(**reward_args(standstill_time=t, delta_root_x=0))
               for t in (0.10, 0.14, 0.3)]
    assert rewards[0]['total_reward'] <= 0
    assert all(b['total_reward'] <= a['total_reward'] for a, b in zip(rewards, rewards[1:]))
    for dx, expected in [(0.003, 0.12), (-0.003, -0.24), (0, 0), (np.nan, 0)]:
        args = reward_args(base_lin_vel=np.array([0.15, 0, 0]), delta_root_x=dx)
        scalar, terms = compute_rewards(**args, return_components=True)
        batch_args = {k: np.stack([v, v]) if isinstance(v, np.ndarray) else v for k, v in args.items()}
        batch, batch_terms = compute_rewards(**batch_args, return_components=True)
        assert scalar['forward_progress'] == pytest.approx(expected)
        assert scalar['total_reward'] == pytest.approx(sum(terms.values()))
        np.testing.assert_allclose(batch['total_reward'], scalar['total_reward'], rtol=1e-6)
        np.testing.assert_allclose(sum(batch_terms.values()), batch['total_reward'])
    for mode in (LocomotionMode.SIT, LocomotionMode.LIE):
        r = compute_rewards(**reward_args(cmd_mode=mode, standstill_time=1, double_support_time=1, delta_root_x=0.01))
        assert r['standstill_penalty'] == r['double_support_penalty'] == r['forward_progress'] == 0
    r = compute_rewards(**reward_args(cmd_lin_vel=np.zeros(3), standstill_time=1, double_support_time=1))
    assert r['standstill_penalty'] == r['double_support_penalty'] == 0


@pytest.mark.parametrize('dt,steps', [(0.02, 2), (0.01, 4), (0.05, 2)])
def test_gait_hysteresis_rejects_chatter_and_tracks_same_foot(dt, steps):
    from training.run_parallel_sim import update_gait_hysteresis_state

    state = dict(standstill_time=np.array([0.2]), progress_streak=np.zeros(1, dtype=int),
                 double_support_time=np.array([0.22]), swing_clear_time=np.zeros((1, 2)),
                 swing_clear_steps=np.zeros((1, 2), dtype=int))

    def step(contact=(False, True), clearance=(0.02, 0), vx=0, dx=0, active=True):
        return update_gait_hysteresis_state(np.array([active]), np.array([vx]), np.array([dx]),
            np.array([contact]), np.array([clearance]), **state, dt=dt)

    assert not step(clearance=(0.002, 0.1)).any()  # support foot cannot qualify
    assert not step().any()
    assert not step(contact=(True, False), clearance=(0, 0.02)).any()
    assert not step(contact=(False, False), clearance=(0.1, 0.1)).any()
    assert state['double_support_time'][0] == pytest.approx(0.22)
    for _ in range(steps - 1):
        assert not step().any()
    assert step()[0, 0]
    assert state['double_support_time'][0] == 0
    dwell = state['standstill_time'][0]
    step(vx=0.1, dx=0.002)
    assert state['standstill_time'][0] == dwell
    step(vx=0.045)
    step(vx=0.1, dx=0.002)
    assert state['standstill_time'][0] == dwell
    step(vx=0.1, dx=0.002)
    assert state['standstill_time'][0] == 0
    step(active=False)
    assert not any(np.any(v) for v in state.values())


@pytest.mark.parametrize('model', ['models/mjcf/microduck_8dof.xml', 'duck_v2/cyberduck_v2.xml'])
def test_sole_clearance_geometry_and_reset_isolation(model):
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner(model, num_envs=2, num_threads=1)
    try:
        d = runner.datas[0]
        gids = [runner.left_foot_gid, runner.right_foot_gid]
        # Synthetic world geometry isolates orientation and terrain from unstable physics.
        for g in gids:
            d.geom_xmat[g] = np.eye(3).ravel()
            d.geom_xpos[g] = [0, 0, runner.model.geom_size[g, 2]]
        np.testing.assert_allclose(runner._compute_sole_clearance()[0], 0, atol=1e-6)
        for g in gids:
            d.geom_xpos[g, 2] += 0.02
        np.testing.assert_allclose(runner._compute_sole_clearance()[0], 0.02, atol=1e-6)
        runner.standstill_time[:] = [0.1, 0.8]
        runner.swing_clear_steps[:] = 3
        runner.reset_env(0)
        assert runner.standstill_time[1] == 0.8
        assert runner.swing_clear_steps[1, 0] == 3
        assert runner.standstill_time[0] == 0
        assert runner.prev_root_x[0] == runner.datas[0].qpos[0]
    finally:
        runner.close()


@pytest.mark.parametrize('fast', [False, True])
def test_terminal_reward_diagnostics_precede_autoreset(fast):
    from training.run_parallel_sim import ParallelSimRunner

    runner = ParallelSimRunner('duck_v2/cyberduck_v2.xml', num_envs=1, max_episode_steps=1, num_threads=1)
    try:
        runner.datas[0].qpos[0] = 1.2
        runner.prev_root_x[0] = 1.2
        mujoco.mj_forward(runner.model, runner.datas[0])
        _, rewards, dones, info = runner.step(np.zeros((1, 8)), fast=fast)
        assert dones[0]
        terms = info['reward_components'] if fast else info[0]['reward_components']
        assert sum(terms.values()) == pytest.approx(float(rewards[0]), abs=1e-5)
        distance = info['terminal_distances_x'][0] if fast else info[0]['distance_x']
        assert distance > 1
        assert runner.prev_root_x[0] == runner.datas[0].qpos[0] == 0
        _, _, _, info2 = runner.step_arrays(np.zeros((1, 8)))
        assert abs(info2['delta_root_x_mean']) < 0.02
    finally:
        runner.close()


@pytest.mark.parametrize('latent', [0., 5., -5., 10., -10., 20., -20.])
def test_squashed_latent_roundtrip_and_gradients(latent):
    from scripts.train_ppo import ActorCritic

    ac = ActorCritic(obs_dim=3, action_dim=2, hidden_dim=8, squash_actions=True)
    obs = torch.zeros(4, 3)
    with torch.no_grad():
        ac.actor[-1].bias.fill_(latent)
    a, lp, entropy, value, u = ac.get_action_and_value(obs, return_pre_tanh=True)
    _, lp2, _, _ = ac.get_action_and_value(obs, action=a.detach(), pre_tanh_action=u.detach())
    assert torch.isfinite(a).all() and (a.abs() <= 1).all()
    torch.testing.assert_close(lp, lp2)
    torch.testing.assert_close((lp2-lp.detach()).exp(), torch.ones_like(lp))
    (-lp2.mean() + value.square().mean() - 0.01 * entropy.mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ac.parameters())
    torch.testing.assert_close(ac.forward_actor(obs), ac.actor(obs).tanh())
    if (a.abs() == 1).any():
        with pytest.raises(ValueError, match='pre_tanh'):
            ac.get_action_and_value(obs, action=a)


def train_small(tmp_path, **kwargs):
    from scripts.train_ppo import train_ppo
    opts = dict(model_path='duck_v2/cyberduck_v2.xml', num_envs=2, rollout_steps=4,
                num_epochs=1, mini_batch_size=4, max_iterations=1, device='cpu', verbose=False,
                save_path=tmp_path/'run_ppo.pt', best_save_path=tmp_path/'run_best.pt', save_interval=1)
    opts.update(kwargs)
    return train_ppo(**opts)


@pytest.mark.parametrize('distance', [0.01, np.nan, np.inf, None])
def test_best_rejects_invalid_or_noise_only_candidate(monkeypatch, tmp_path, distance):
    import scripts.train_ppo as ppo
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs['stage'])
        return distance, 1.0
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', evaluate)
    monkeypatch.setattr(ppo.ParallelSimRunner, 'get_mean_distance_x', lambda self: 0.85)
    result = train_small(tmp_path)
    assert calls == [0]
    assert result['best_checkpoint_path'] is None
    assert not (tmp_path/'run_best.pt').exists()  # includes final-save route
    assert (tmp_path/'run_latest.pt').exists()
    assert result['metrics']['best_rejection_reason']


def test_checkpoint_accept_resume_reject_and_latent_buffer(monkeypatch, tmp_path):
    import scripts.train_ppo as ppo
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', lambda **kw: (0.30, 1.0))
    original = ppo.ActorCritic.get_action_and_value
    sampled = []
    evaluated = []
    def capture(self, obs, action=None, **kw):
        out = original(self, obs, action, **kw)
        if action is None:
            sampled.append(out[0].detach().numpy().copy())
        elif self.squash_actions:
            assert kw['pre_tanh_action'] is not None
            torch.testing.assert_close(action, kw['pre_tanh_action'].tanh())
            evaluated.append(True)
        return out
    monkeypatch.setattr(ppo.ActorCritic, 'get_action_and_value', capture)
    delivered = []
    step = ppo.ParallelSimRunner.step_arrays
    def capture_step(self, actions):
        delivered.append(actions.copy())
        return step(self, actions)
    monkeypatch.setattr(ppo.ParallelSimRunner, 'step_arrays', capture_step)
    result = train_small(tmp_path)
    assert evaluated and len(sampled) == len(delivered) == 4
    for sample, sent in zip(sampled, delivered):
        np.testing.assert_array_equal(sample, sent)
    best = tmp_path/'run_best.pt'
    before = best.read_bytes()
    for name in ('run_ppo.pt', 'run_latest.pt', 'run_best.pt'):
        ckpt = torch.load(tmp_path/name, weights_only=False)
        assert ckpt['squash_actions'] is True
        assert ckpt['model_name'] == 'cyberduck_v2.xml'
        assert ckpt['best_det_distance_x'] == 0.3
        assert ckpt['best_selection']['protocol'] == ckpt['evaluation_protocol']
    assert result['metrics']['reward_components']
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', lambda **kw: (0.01, 20.0))
    train_small(tmp_path, resume=True)
    assert best.read_bytes() == before
    ckpt = torch.load(tmp_path/'run_ppo.pt', weights_only=False)
    assert ckpt['best_selection']['det_distance_x'] == 0.3


@pytest.mark.parametrize('options', [{'dry_run': True}, {'stand_first': True}])
def test_standing_and_dry_runs_never_create_walking_best(monkeypatch, tmp_path, options):
    import scripts.train_ppo as ppo
    def unexpected(**kw):
        pytest.fail('Standing/dry-run must not qualify a walking candidate')
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', unexpected)
    result = train_small(tmp_path, **options)
    assert result['best_checkpoint_path'] is None
    assert not (tmp_path/'run_best.pt').exists()


def test_legacy_resume_preserves_transform_and_transfer_enables_squash(monkeypatch, tmp_path):
    import scripts.train_ppo as ppo
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', lambda **kw: (0.1, 1.0))
    train_small(tmp_path, dry_run=True)
    ckpt = torch.load(tmp_path/'run_ppo.pt', weights_only=False)
    ckpt.pop('squash_actions')
    torch.save(ckpt, tmp_path/'run_ppo.pt')
    train_small(tmp_path, resume=True, dry_run=True)
    assert torch.load(tmp_path/'run_ppo.pt', weights_only=False)['squash_actions'] is False
    train_small(tmp_path/'transfer', load_checkpoint=tmp_path/'run_ppo.pt', dry_run=True)
    assert torch.load(tmp_path/'transfer/run_ppo.pt', weights_only=False)['squash_actions'] is True


def test_deterministic_evaluation_counts_terminal_step_and_restores_state():
    from scripts.train_ppo import ActorCritic, RunningMeanStd, evaluate_deterministic_rollout
    ac = ActorCritic(obs_dim=3, action_dim=2, hidden_dim=8, squash_actions=True)
    rms = RunningMeanStd(3)
    original_rng = np.random.default_rng(123)
    runner = SimpleNamespace(num_envs=1, rng=original_rng, friction_range=(1, 2),
        curriculum_stage=0, datas=[SimpleNamespace(qpos=np.array([0.0]))], commands=np.zeros((1, 3)))
    runner.reset = lambda: np.zeros((1, 3))
    runner._extract_obs = lambda i: np.zeros(3)
    count = []
    def step(actions):
        count.append(1)
        return np.zeros((1, 3)), np.array([2.0]), np.array([True]), {'terminal_distances_x': np.array([0.4])}
    runner.step_arrays = step
    torch_rng = torch.get_rng_state().clone()
    dist, reward = evaluate_deterministic_rollout(ac, rms, 'unused', 0, eval_runner=runner)
    assert dist == pytest.approx(0.4) and reward == 2
    assert len(count) == 3 and ac.training
    assert runner.rng is original_rng and runner.friction_range == (1, 2)
    assert rms.count == 1e-4
    torch.testing.assert_close(torch.get_rng_state(), torch_rng)


def policy_checkpoint(squash=True, obs_dim=42, hidden_dim=32):
    from scripts.train_ppo import ActorCritic, RunningMeanStd
    model = ActorCritic(obs_dim=obs_dim, action_dim=8, hidden_dim=hidden_dim, squash_actions=squash)
    rms = RunningMeanStd(obs_dim, clip=2.0)
    rms.mean[:] = 0.3
    rms.var[:] = 0.4
    ckpt = dict(model_state_dict=model.state_dict(), obs_dim=obs_dim, action_dim=8,
                hidden_dim=hidden_dim, squash_actions=squash, obs_rms=rms.state_dict(),
                model_path='duck_v2/cyberduck_v2.xml', model_name='cyberduck_v2.xml')
    return model, rms, ckpt


@pytest.mark.parametrize('bad', [None, {}, {'mean': np.zeros(41)}, {'mean': np.zeros(43)},
    {'mean': np.zeros((1, 42))}, {'mean': np.full(42, np.nan)}, {'var': -np.ones(42)},
    {'var': np.full(42, np.inf)}, {'epsilon': -1}, {'min_std': 0}, {'clip': np.nan}, {'count': -1}])
def test_rms_invalid_rejected_without_mutation(tmp_path, bad):
    from scripts.export_policy import apply_obs_norm_to_actor
    from scripts.interactive_viewer import load_viewer_policy
    model, _, ckpt = policy_checkpoint()
    if bad is None or bad == {}:
        ckpt['obs_rms'] = bad
    else:
        ckpt['obs_rms'].update(bad)
    before = {k: v.clone() for k, v in model.actor.state_dict().items()}
    with pytest.raises(ValueError, match='obs_rms'):
        apply_obs_norm_to_actor(model.actor, ckpt)
    for k, v in model.actor.state_dict().items():
        torch.testing.assert_close(v, before[k])
    path = tmp_path/'bad.pt'
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='obs_rms'):
        load_viewer_policy(path)


@pytest.mark.parametrize('squash', [False, True])
def test_viewer_onnx_normalization_transform_parity(tmp_path, squash):
    import onnxruntime as ort
    from scripts.export_policy import export_onnx, load_actor_from_checkpoint, apply_obs_norm_to_actor
    from scripts.interactive_viewer import load_viewer_policy, policy_step
    model, rms, ckpt = policy_checkpoint(squash)
    path = tmp_path/'unrelated.pt'
    torch.save(ckpt, path)
    policy = load_viewer_policy(path)
    onnx = export_onnx(path, tmp_path/'policy.onnx')
    session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
    for raw in (np.zeros(42, dtype=np.float32), np.linspace(-100, 100, 42, dtype=np.float32)):
        with torch.no_grad():
            expected = model.forward_actor(torch.as_tensor(rms.normalize(raw), dtype=torch.float32)).numpy()
        np.testing.assert_allclose(policy_step(policy, raw), np.clip(expected, -1, 1), atol=2e-6)
        np.testing.assert_allclose(session.run(None, {'obs': raw[None]})[0][0], expected, atol=2e-6)
    before = policy[0].weight.clone()
    apply_obs_norm_to_actor(policy, ckpt)
    torch.testing.assert_close(policy[0].weight, before)
    direct = load_actor_from_checkpoint(model)
    assert getattr(direct, '_squash_actions', False) == squash


def test_viewer_model_resolution_explicitness_and_metadata(tmp_path):
    from scripts.interactive_viewer import resolve_viewer_model_path, build_arg_parser
    _, _, ckpt = policy_checkpoint()
    path = tmp_path/'unrelated.pt'
    torch.save(ckpt, path)
    assert build_arg_parser().parse_args([]).model is None
    assert Path(resolve_viewer_model_path(None, path)).name == 'cyberduck_v2.xml'
    with pytest.raises(ValueError, match='model'):
        resolve_viewer_model_path('models/mjcf/microduck_8dof.xml', path)
    ckpt.pop('model_path'); ckpt.pop('model_name')
    torch.save(ckpt, path)
    assert Path(resolve_viewer_model_path('models/mjcf/microduck_8dof.xml', path)).name == 'microduck_8dof.xml'
    assert Path(resolve_viewer_model_path(None, path)).name == 'microduck_8dof.xml'
    named = tmp_path/'cyberduck_v2_best.pt'
    torch.save(ckpt, named)
    assert Path(resolve_viewer_model_path(None, named)).name == 'cyberduck_v2.xml'
    ambiguous = tmp_path/'experiment_v2.pt'
    torch.save(ckpt, ambiguous)
    assert Path(resolve_viewer_model_path(None, ambiguous)).name == 'microduck_8dof.xml'
    ckpt['model_path'] = 'missing/model.xml'
    torch.save(ckpt, named)
    with pytest.raises((ValueError, FileNotFoundError), match='model'):
        resolve_viewer_model_path(None, named)


@pytest.mark.parametrize('squash', [False, True])
def test_generated_c_applies_output_transform_before_quantization(tmp_path, squash):
    import subprocess
    from scripts.export_policy import export_esp32_headers
    from scripts.train_ppo import ActorCritic
    model = ActorCritic(hidden_dim=256, squash_actions=squash)
    with torch.no_grad():
        for p in model.actor.parameters():
            p.zero_()
        # Deliberately zero weights exposes bias-only quantization overflow too.
        model.actor[-1].bias.copy_(torch.tensor([0.2, -0.2, 0.5, -0.5, 1., -1., 2., -2.]))
    weights, config = export_esp32_headers(model, tmp_path/'weights.h', tmp_path/'config.h')
    source = tmp_path/'run.cpp'
    source.write_text('#include <cstdio>\n#include "config.h"\n#include "weights.h"\n'
                      'int main(){int8_t obs[42]={0}, a[8]; evaluate_policy_int8(obs,a);'
                      'for(int i=0;i<8;i++) printf("%d ", (int)a[i]);}\n')
    result = subprocess.run(['g++', '-std=c++17', str(source), '-o', str(tmp_path/'run.exe')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(tmp_path/'run.exe')], capture_output=True, text=True, check=True)
    actual = np.fromstring(result.stdout, sep=' ')
    with torch.no_grad():
        expected = model.forward_actor(torch.zeros(1, 42)).numpy()[0]
    expected = np.clip(np.round(expected * 127), -128, 127)
    np.testing.assert_allclose(actual, expected, atol=1)


def test_pytorch_viewing_has_no_onnx_dependency(monkeypatch, tmp_path):
    import builtins
    import importlib
    import scripts.export_policy as exporter
    from scripts.interactive_viewer import load_viewer_policy
    _, _, ckpt = policy_checkpoint()
    path = tmp_path/'policy.pt'
    torch.save(ckpt, path)
    original_import = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'onnx' or name.startswith('onnx.'):
            raise ImportError('ONNX intentionally unavailable')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    importlib.reload(exporter)
    assert load_viewer_policy(path)._obs_norm_applied


def test_deterministic_evaluation_repeats_same_episode_seeds():
    from scripts.train_ppo import ActorCritic, RunningMeanStd, evaluate_deterministic_rollout
    from training.run_parallel_sim import ParallelSimRunner
    model = ActorCritic(squash_actions=True)
    runner = ParallelSimRunner('duck_v2/cyberduck_v2.xml', num_envs=1)
    rms = RunningMeanStd(42)
    try:
        kwargs = dict(model=model, obs_rms=rms, model_path=runner.model_path, stage=0,
                      eval_runner=runner, max_steps=5)
        first = evaluate_deterministic_rollout(**kwargs)
        second = evaluate_deterministic_rollout(**kwargs)
        np.testing.assert_allclose(first, second, atol=1e-7)
    finally:
        runner.close()


def test_shared_checkpoint_consumers_keep_normalization_and_squash(tmp_path):
    from scripts.evaluate_models_sim import load_actor
    from scripts.export_policy import load_actor_from_checkpoint
    from scripts.pretrain_bc import collect_teacher_demonstrations
    model, rms, ckpt = policy_checkpoint()
    path = tmp_path/'policy.pt'
    torch.save(ckpt, path)
    raw = np.full((1, 42), 50, dtype=np.float32)
    with torch.no_grad():
        expected = model.forward_actor(torch.as_tensor(rms.normalize(raw), dtype=torch.float32)).numpy()
        actual = load_actor(path)(torch.from_numpy(raw)).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    import scripts.pretrain_bc as bc
    # Exercise the teacher policy callback without spending physics steps.
    from unittest.mock import patch
    def collect(runner, policy_fn, steps, gamma):
        return policy_fn(raw, 0)
    with patch.object(bc, '_collect_rollouts', collect):
        teacher_actions = collect_teacher_demonstrations(None, load_actor_from_checkpoint(path), num_steps=1)
    np.testing.assert_allclose(teacher_actions, expected, atol=2e-6)


def test_viewer_rejects_invalid_actor_dimensions_and_corrupt_hot_reload(tmp_path, capsys):
    from scripts.interactive_viewer import load_viewer_policy, _maybe_hot_reload_policy
    _, _, ckpt = policy_checkpoint(obs_dim=41)
    path = tmp_path/'policy.pt'
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match='dimensions'):
        load_viewer_policy(path)
    _, _, ckpt = policy_checkpoint()
    torch.save(ckpt, path)
    good = load_viewer_policy(path)
    ckpt['obs_rms'] = None
    torch.save(ckpt, path)
    policy, _, _ = _maybe_hot_reload_policy(path, good, 0, 0, now=100)
    assert policy is good
    assert 'Keeping last valid policy' in capsys.readouterr().out


def test_direct_module_onnx_transform_and_export_model_metadata(tmp_path):
    import onnxruntime as ort
    from scripts.export_policy import export_onnx
    from scripts.interactive_viewer import resolve_viewer_model_path
    model, rms, ckpt = policy_checkpoint()
    with torch.no_grad():
        model.actor[-1].bias.fill_(2)
    path = export_onnx(model, tmp_path/'direct.onnx')
    raw = np.zeros((1, 42), dtype=np.float32)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        expected = model.forward_actor(torch.from_numpy(raw)).numpy()
    np.testing.assert_allclose(session.run(None, {'obs': raw})[0], expected, atol=1e-6)
    ckpt['model_path'] = 'duck_v2\\cyberduck_v2.xml'
    path = export_onnx(ckpt, tmp_path/'metadata.onnx')
    assert Path(resolve_viewer_model_path(None, path)).name == 'cyberduck_v2.xml'


def test_best_selection_protocol_and_stage_order():
    from scripts.train_ppo import select_deterministic_candidate
    previous = dict(stage=1, det_distance_x=0.3, det_reward=2, protocol={'version': 1})
    assert not select_deterministic_candidate(dict(previous, stage=0, det_distance_x=1), previous)[0]
    assert not select_deterministic_candidate(dict(previous, stage=2, det_distance_x=0.01), previous)[0]
    assert select_deterministic_candidate(dict(previous, stage=2, det_distance_x=0.2), previous)[0]
    assert select_deterministic_candidate(dict(previous, det_reward=3), previous)[0]
    assert not select_deterministic_candidate(dict(previous, protocol={'version': 2}), previous)[0]


def test_failed_best_write_preserves_existing_file(monkeypatch, tmp_path):
    import scripts.train_ppo as ppo
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', lambda **kw: (0.3, 1.0))
    train_small(tmp_path)
    before = (tmp_path/'run_best.pt').read_bytes()
    save = ppo._atomic_torch_save
    def fail(payload, path):
        if Path(path).name == 'run_best.pt':
            raise OSError('simulated disk failure')
        return save(payload, path)
    monkeypatch.setattr(ppo, '_atomic_torch_save', fail)
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', lambda **kw: (0.4, 1.0))
    with pytest.raises(OSError, match='disk failure'):
        train_small(tmp_path, resume=True)
    assert (tmp_path/'run_best.pt').read_bytes() == before


@pytest.mark.parametrize('yaw,world_vx,dx,valid', [(0, 0.15, 0.003, True),
    (np.pi/3, 0.15, 0.003, True), (0, 0, 0.2, False), (0, 0.15, -1.2, False),
    (0, -0.15, -0.003, True)])
def test_progress_consistency_uses_world_frame(monkeypatch, yaw, world_vx, dx, valid):
    from training.run_parallel_sim import ParallelSimRunner
    runner = ParallelSimRunner('duck_v2/cyberduck_v2.xml', num_envs=1, num_threads=1)
    try:
        d = runner.datas[0]
        d.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        d.qvel[0] = world_vx
        runner.commands[0] = [0.2, 0, 0]
        mujoco.mj_forward(runner.model, d)
        def scripted_physics(indices, targets):
            d.qpos[0] += dx
            mujoco.mj_forward(runner.model, d)
        monkeypatch.setattr(runner, '_step_subrange', scripted_physics)
        _, _, _, info = runner.step_arrays(np.zeros((1, 8)))
        assert info['invalid_displacement_count'] == (0 if valid else 1)
        assert info['delta_root_x_mean'] == pytest.approx(dx if valid else 0, abs=1e-6)
    finally:
        runner.close()


def test_standing_transition_is_evaluated_on_next_walking_rollout(monkeypatch, tmp_path):
    import scripts.train_ppo as ppo
    monkeypatch.setattr(ppo.ParallelSimRunner, 'get_mean_episode_length', lambda self: 400.)
    calls = []
    def evaluate(**kw):
        calls.append(kw['stage'])
        return 0.3, 1.0
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', evaluate)
    result = train_small(tmp_path, stand_first=True, max_iterations=2)
    assert calls == [0]
    ckpt = torch.load(result['best_checkpoint_path'], weights_only=False)
    assert ckpt['iterations'] == 2
    assert ckpt['metrics']['evaluation_mode'] == 'walk'


def test_checkpoint_path_alias_cannot_bypass_selection(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        train_small(tmp_path, best_save_path=tmp_path/'run_ppo.pt')
    with pytest.raises(ValueError, match='Transfer'):
        train_small(tmp_path, load_checkpoint=tmp_path/'run_ppo.pt')


def test_resume_with_corrupt_best_preserves_it_until_qualified(monkeypatch, tmp_path):
    import scripts.train_ppo as ppo
    train_small(tmp_path, dry_run=True)
    best = tmp_path/'run_best.pt'
    best.write_bytes(b'corrupt non-pickle data')
    monkeypatch.setattr(ppo, 'evaluate_deterministic_rollout', lambda **kw: (0.01, 1))
    result = train_small(tmp_path, resume=True)
    assert result['best_checkpoint_path'] is None
    assert best.read_bytes() == b'corrupt non-pickle data'


def test_sole_clearance_accounts_for_rotation_and_raised_terrain():
    from training.run_parallel_sim import ParallelSimRunner
    runner = ParallelSimRunner('duck_v2/cyberduck_v2.xml', num_envs=1)
    try:
        d = runner.datas[0]
        gid = runner.left_foot_gid
        angle = np.pi/6
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        rotated = runner._sole_pts_local[0] @ rotation.T
        terrain = runner.terrain_box_gids[0]
        runner.model.geom_pos[terrain] = [0, 0, 0.05]
        runner.model.geom_size[terrain] = [0.5, 0.5, 0.05]
        runner.model.geom_contype[terrain] = 1
        d.geom_xmat[gid] = rotation.ravel()
        d.geom_xpos[gid] = [0, 0, 0.10 - rotated[:, 2].min()]
        assert runner._compute_sole_clearance()[0, 0] == pytest.approx(0, abs=1e-6)
        d.geom_xpos[gid, 2] += 0.01
        assert runner._compute_sole_clearance()[0, 0] == pytest.approx(0.01, abs=1e-6)
        runner._sole_pts_local = None
        np.testing.assert_array_equal(runner._compute_sole_clearance(), 0)
    finally:
        runner.close()
