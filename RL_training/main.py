import datetime
import os
import sys
from argparse import ArgumentParser

from pathlib import Path

import torch


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Read Configs ---
    agent_config = config["agent_config"]
    navigation_config = config["navigation_config"]
    env_config = config["env_config"]

    encoder_path = Path(agent_config["encoder_path"])
    encoder_path = encoder_path.parent / (encoder_path.stem + "_" + str(navigation_config["use_transformer"]) + encoder_path.suffix)

    # Setup environment
    if not args.precomputed:
        env = ThorEnv(render=env_config["render"], rho=env_config["rho"], max_actions=agent_config["num_steps"])
    else:
        env = PrecomputedThorEnv(rho=env_config["rho"], max_actions=agent_config["num_steps"])

    # Load agent from encoder & policy weights
    if agent_config["name"] == "reinforce":
        agent = ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    elif agent_config["name"] == "a2c":
        agent = A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    else:
        raise Exception("Unknown agent")

    agent.load_weights(encoder_path=encoder_path, device=device)

    # RL training runner
    runner = RLTrainRunner(env=env, agent=agent, device=device)
    runner.run()
    env.close()
    print("[INFO] Training completed.")

    # --- Save final model ---
    if args.save_model:
        save_folder = Path("RL_training") / "runs" / "model_weights"
        save_folder.mkdir(exist_ok=True)
        run_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fle_name = f"{run_start}_{agent_config["name"]}_agent.pth"
        agent.save_model(str(save_folder), file_name=fle_name)


def set_working_directory():
    desired_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working director:", os.getcwd())


if __name__ == "__main__":
    set_working_directory()

    from components.agents.a2c_agent import A2CAgent
    from components.agents.reinforce_agent import ReinforceAgent
    from components.environments.thor_env import ThorEnv
    from components.environments.precomputed_thor_env import PrecomputedThorEnv
    from RL_training.runner.rl_train_runner import RLTrainRunner
    from components.utils.utility_functions import read_config, set_seeds

    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str, help="Path to the configuration files.")
    parser.add_argument("--save_model", action="store_true", help="Save model weights.")
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed environment.")
    args = parser.parse_args()

    # Iterate over each configuration file in args.conf_path
    conf_files = Path(args.conf_path).rglob("*.json")
    for conf_file in conf_files:
        print(f"[INFO] Running with configuration: {conf_file}")
        conf = read_config(conf_file)
        set_seeds(conf["seed"])
        main(conf)
