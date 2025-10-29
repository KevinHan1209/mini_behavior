#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import numpy as np
import torch

# Ensure project root is on sys.path to allow `import algorithms.NovelD_PPO`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Prefer gymnasium if available; fall back to gym
try:
    import gymnasium as gym
    GYMNASIUM = True
except Exception:
    import gym
    GYMNASIUM = False

# Optional deps
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

try:
    import matplotlib
    # Force a non-interactive backend for headless environments
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    MPL_AVAILABLE = True
    MPL_IMPORT_ERROR = None
except Exception as e:
    plt = None
    patches = None
    MPL_AVAILABLE = False
    MPL_IMPORT_ERROR = e

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    imageio = None
    IMAGEIO_AVAILABLE = False

from algorithms.NovelD_PPO import NovelD_PPO


def reset_env(env, seed=None):
    try:
        if seed is not None:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()
        return obs, info
    except Exception:
        # Older gym API
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        obs = env.reset()
        info = {}
        return obs, info


def step_env(env, action):
    try:
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    except ValueError:
        # Older gym API
        obs, reward, done, info = env.step(action)
        return obs, float(reward), bool(done), info


def evaluate_agent(ppo: NovelD_PPO, env_id: str, episodes: int = 5, render: bool = False, return_returns: bool = False):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = reset_env(env)
        ep_ret = 0.0
        done = False
        while not done:
            if render:
                try:
                    env.render()
                except Exception:
                    pass
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
            action_np = action.detach().cpu().numpy()
            # For single Discrete action spaces, squeeze and cast to int
            if len(ppo.action_dims) == 1:
                action_env = int(action_np.squeeze())
            else:
                # MultiDiscrete (not used for CartPole)
                action_env = action_np[0]
            obs, reward, done, _ = step_env(env, action_env)
            ep_ret += reward
        returns.append(ep_ret)
    env.close()
    avg_return = float(np.mean(returns)) if returns else 0.0
    if return_returns:
        return avg_return, returns
    return avg_return


def get_frame(env):
    try:
        frame = env.render()
        if frame is not None:
            return frame
    except Exception:
        pass
    # Fallback for older gym
    try:
        return env.render(mode="rgb_array")
    except Exception:
        return None


def record_policy_video(ppo: NovelD_PPO, env_id: str, max_steps: int = 500, fps: int = 20, out_dir: str = None, fmt: str = "mp4"):
    if not IMAGEIO_AVAILABLE:
        print("[viz] imageio not available. Install with `pip install imageio`.")
        return None, None
    # Create env and step while grabbing frames
    try:
        env = gym.make(env_id, render_mode="rgb_array") if GYMNASIUM else gym.make(env_id)
    except Exception:
        env = gym.make(env_id)
    obs, _ = reset_env(env)
    frames = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
        action_np = action.detach().cpu().numpy()
        action_env = int(action_np.squeeze()) if len(ppo.action_dims) == 1 else action_np[0]
        obs, reward, done, _ = step_env(env, action_env)
        frame = get_frame(env)
        if frame is not None:
            frames.append(frame)
        steps += 1
    env.close()
    if not frames:
        print("[viz] No frames captured from env-rendered video.")
        return None, None

    # Save video
    out_dir = out_dir or os.path.join(REPO_ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "ppo_cartpole_episode")
    if fmt == "gif":
        out_path = base + ".gif"
        try:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[viz] Saved env-rendered {fmt} with {len(frames)} frames to {out_path}")
            return out_path, frames
        except Exception as e:
            print(f"[viz] Could not write GIF: {e}")
            return None, frames
    elif fmt == "mp4":
        out_path = base + ".mp4"
        try:
            writer = imageio.get_writer(out_path, fps=fps)
            for f in frames:
                writer.append_data(f)
            writer.close()
            print(f"[viz] Saved env-rendered {fmt} with {len(frames)} frames to {out_path}")
            return out_path, frames
        except Exception as e:
            print(f"[viz] Could not write MP4: {e}. Try `pip install imageio-ffmpeg`.")
            return None, frames
    else:
        print(f"[viz] Unsupported video format: {fmt}")
        return None, frames


def _draw_cartpole_frame(ax, x, theta, x_threshold, width_px=640, height_px=360):
    ax.clear()
    ax.set_xlim(-x_threshold * 1.2, x_threshold * 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')

    # Ground
    ax.plot([-x_threshold * 1.2, x_threshold * 1.2], [0, 0], color='black', linewidth=2)

    # Cart parameters (in plot units)
    cart_w = 0.4
    cart_h = 0.2
    cart_y = cart_h / 2

    # Pole parameters
    pole_len = 1.0  # just for drawing (not exact physical length)

    # Draw cart as rectangle centered at (x, cart_y)
    cart = patches.Rectangle((x - cart_w / 2, 0), cart_w, cart_h, color='#4C72B0')
    ax.add_patch(cart)

    # Draw pole as a line from cart top center
    cart_top_x = x
    cart_top_y = cart_h
    pole_x_end = cart_top_x + pole_len * np.sin(theta)
    pole_y_end = cart_top_y + pole_len * np.cos(theta)
    ax.plot([cart_top_x, pole_x_end], [cart_top_y, pole_y_end], color='#DD8452', linewidth=3)

    # Draw axle
    ax.add_patch(patches.Circle((cart_top_x, cart_top_y), 0.02, color='k'))


def record_cartpole_matplotlib_video(ppo: NovelD_PPO, env_id: str, max_steps: int = 500, fps: int = 20, out_dir: str = None, fmt: str = "mp4"):
    if not MPL_AVAILABLE:
        print(f"[viz] Matplotlib not available: {MPL_IMPORT_ERROR}")
        return None
    if not IMAGEIO_AVAILABLE:
        print("[viz] imageio not available. Install with `pip install imageio`.")
        return None

    try:
        # Query x_threshold from a fresh env
        try:
            probe_env = gym.make(env_id)
            x_threshold = float(getattr(probe_env.unwrapped, 'x_threshold', 2.4))
            probe_env.close()
        except Exception:
            x_threshold = 2.4

        # Run an episode and collect obs
        env = gym.make(env_id)
        obs, _ = reset_env(env)
        frames = []
        done = False
        steps = 0

        # Prepare a figure once and reuse
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=100)

        while not done and steps < max_steps:
            # Draw from observation (CartPole obs = [x, x_dot, theta, theta_dot])
            try:
                x = float(obs[0])
                theta = float(obs[2])
            except Exception:
                # Fallback to env.unwrapped.state if obs missing
                st = getattr(env.unwrapped, 'state', None)
                if st is None:
                    break
                x = float(st[0])
                theta = float(st[2])

            _draw_cartpole_frame(ax, x, theta, x_threshold)
            fig.canvas.draw()
            # Convert figure to image
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape((h, w, 3))
            frames.append(img)

            # Step with policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=ppo.device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _, _ = ppo.agent.get_action_and_value(obs_tensor)
            action_np = action.detach().cpu().numpy()
            action_env = int(action_np.squeeze()) if len(ppo.action_dims) == 1 else action_np[0]
            obs, _, done, _ = step_env(env, action_env)
            steps += 1

        plt.close(fig)
        env.close()

        if not frames:
            print("[viz] No frames collected for Matplotlib visualization.")
            return None

        out_dir = out_dir or os.path.join(REPO_ROOT, 'outputs')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, 'ppo_cartpole_matplotlib')
        if fmt == 'gif':
            out_path = base + '.gif'
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"[viz] Saved Matplotlib {fmt} with {len(frames)} frames to {out_path}")
            return out_path
        elif fmt == 'mp4':
            out_path = base + '.mp4'
            try:
                writer = imageio.get_writer(out_path, fps=fps)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()
                print(f"[viz] Saved Matplotlib {fmt} with {len(frames)} frames to {out_path}")
                return out_path
            except Exception as e:
                print(f"[viz] Could not write MP4: {e}. Try `pip install imageio-ffmpeg`.")
                return None
        else:
            print(f"[viz] Unsupported video format: {fmt}")
            return None
    except Exception as e:
        print(f"[viz] Matplotlib video error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="PPO optimizer sanity check on CartPole")
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=50, help="number of PPO updates (total_timesteps = num_envs*num_steps*updates)")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pre_episodes", type=int, default=5)
    parser.add_argument("--post_episodes", type=int, default=5)
    parser.add_argument("--target_return", type=float, default=150.0, help="success threshold for average return after training")
    parser.add_argument("--render_eval", action="store_true")
    parser.add_argument("--eval_every", type=int, default=5, help="evaluate and log returns every N updates")
    parser.add_argument("--eval_episodes", type=int, default=5, help="episodes per evaluation point")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="PPO-Sanity")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_video", action="store_true", help="record and log a video after training (env-rendered)")
    parser.add_argument("--viz_cartpole", action="store_true", help="record and log a Matplotlib-drawn CartPole video after training")
    parser.add_argument("--video_format", choices=["gif", "mp4"], default="mp4", help="format for saved videos and W&B logging")
    args = parser.parse_args()

    if args.wandb and not WANDB_AVAILABLE:
        print("wandb not installed; proceeding without wandb logging.")
        args.wandb = False

    total_timesteps = args.num_envs * args.num_steps * args.updates

    # Initialize W&B (optional)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "env_id": args.env_id,
                "device": args.device,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "updates": args.updates,
                "total_timesteps": total_timesteps,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "int_coef": 0.0,
                "ext_coef": 1.0,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
            },
        )
        # Define metrics for nicer plots
        try:
            wandb.define_metric("update")
            wandb.define_metric("eval/return", step_metric="update")
        except Exception:
            pass

    # Instantiate NovelD_PPO with wrapper disabled and extrinsic-only optimization
    ppo = NovelD_PPO(
        env_id=args.env_id,
        device=args.device,
        total_timesteps=total_timesteps,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
        int_coef=0.0,
        ext_coef=1.0,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wrap_observations=False,
        norm_adv=True,
        use_intrinsic=False,
    )

    # Set up per-update evaluation callback to log returns vs iteration
    eval_curve = []  # list of (update, avg_return)
    def on_update(update, global_step, agent_ref):
        if args.eval_every > 0 and (update % args.eval_every == 0):
            avg_ret, rets = evaluate_agent(agent_ref, args.env_id, episodes=args.eval_episodes, render=False, return_returns=True)
            eval_curve.append((update, avg_ret))
            print(f"[Eval] Update {update}: avg_return={avg_ret:.2f}")
            if args.wandb:
                wandb.log({
                    "update": update,
                    "eval/return": avg_ret,
                    "eval/returns_hist_iter": wandb.Histogram(rets),
                }, step=update)
    ppo.update_callback = on_update

    print("Evaluating before training...")
    pre_return, pre_returns = evaluate_agent(ppo, args.env_id, episodes=args.pre_episodes, render=args.render_eval, return_returns=True)
    print(f"Average return before training: {pre_return:.2f}")
    if args.wandb:
        # Scalar and histogram of per-episode returns
        wandb.log({
            "eval/pre_return": pre_return,
            "eval/pre_returns_hist": wandb.Histogram(pre_returns),
        })

    print("\nStarting training...")
    ppo.train()

    print("\nEvaluating after training...")
    post_return, post_returns = evaluate_agent(ppo, args.env_id, episodes=args.post_episodes, render=args.render_eval, return_returns=True)
    print(f"Average return after training: {post_return:.2f}")

    improved = post_return > pre_return + 20.0  # require at least 20 point improvement
    meets_target = post_return >= args.target_return

    if args.wandb:
        # Scalars, histograms, and a W&B bar chart for pre vs post
        pre_post_table = wandb.Table(data=[["pre", pre_return], ["post", post_return]], columns=["phase", "avg_return"])
        wandb.log({
            "eval/post_return": post_return,
            "eval/post_returns_hist": wandb.Histogram(post_returns),
            "eval/improvement": post_return - pre_return,
            "eval/returns_bar": wandb.plot.bar(pre_post_table, "phase", "avg_return", title="CartPole Pre vs Post Return"),
            "sanity/improved": improved,
            "sanity/meets_target": meets_target,
        })

    # Optional: record and log a video (env-rendered)
    if args.log_video:
        vid_path, frames = record_policy_video(ppo, args.env_id, max_steps=500, fps=20, fmt=args.video_format)
        if vid_path:
            print(f"Saved evaluation video to {vid_path}")
            if args.wandb:
                wandb.log({"eval/video_env": wandb.Video(vid_path, fps=20, format=args.video_format)})
        else:
            print("Could not record env-rendered video (imageio not available or env rendering unsupported).")

    # Optional: record and log a Matplotlib-drawn CartPole video (robust and not dependent on env.render)
    if args.viz_cartpole:
        viz_path = record_cartpole_matplotlib_video(ppo, args.env_id, max_steps=500, fps=20, fmt=args.video_format)
        if viz_path:
            print(f"Saved Matplotlib CartPole video to {viz_path}")
            if args.wandb:
                wandb.log({"eval/video_matplotlib": wandb.Video(viz_path, fps=20, format=args.video_format)})
        else:
            print("Could not record Matplotlib CartPole video (missing matplotlib/imageio or an error occurred).\n"
                  "To enable, install the extras in your current environment:\n"
                  "  python -m pip install matplotlib imageio imageio-ffmpeg\n"
                  "If running headless, this script now uses the 'Agg' backend automatically.")

    # Optional: simple visualization of pre vs post return
    if MPL_AVAILABLE:
        fig = plt.figure(figsize=(4, 3))
        plt.bar(["pre", "post"], [pre_return, post_return], color=["#9999ff", "#66cc66"]) 
        plt.ylabel("Average Return")
        plt.title("PPO Sanity Check: CartPole (Pre vs Post)")
        plt.tight_layout()
        # Save locally
        out_dir = os.path.join(REPO_ROOT, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, "ppo_sanity_returns.png")
        plt.savefig(fig_path)
        print(f"Saved return plot to {fig_path}")
        if args.wandb:
            wandb.log({"eval/return_plot": wandb.Image(fig)})
        plt.close(fig)

    # Return vs iteration line plot
    if MPL_AVAILABLE and len(eval_curve) > 0:
        updates = [u for u, _ in eval_curve]
        values = [v for _, v in eval_curve]
        fig2 = plt.figure(figsize=(5, 3))
        plt.plot(updates, values, marker="o")
        plt.xlabel("Update")
        plt.ylabel("Average Return")
        plt.title("CartPole Return vs Update")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_dir = os.path.join(REPO_ROOT, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        curve_path = os.path.join(out_dir, "ppo_return_curve.png")
        plt.savefig(curve_path)
        print(f"Saved return-vs-update plot to {curve_path}")
        if args.wandb:
            wandb.log({"eval/return_curve_img": wandb.Image(fig2)})
            # Also log a native W&B line plot if available
            try:
                from wandb import plot
                xs = updates
                ys = [values]
                keys = ["avg_return"]
                wandb.log({"eval/return_curve": wandb.plot.line_series(xs=xs, ys=ys, keys=keys, title="CartPole Return vs Update", xname="update")})
            except Exception:
                pass
        plt.close(fig2)

    print("\nSanity check result:")
    print(f"Improved: {improved} (Δ={post_return - pre_return:.1f}) | Meets target ({args.target_return}): {meets_target}")

    # Exit code semantics: 0 if passing the sanity check, 1 otherwise
    exit_code = 0 if (meets_target or improved) else 1

    if args.wandb:
        # Ensure the wandb run is closed if we opened it
        try:
            wandb.finish()
        except Exception:
            pass

    exit(exit_code)


if __name__ == "__main__":
    main()

