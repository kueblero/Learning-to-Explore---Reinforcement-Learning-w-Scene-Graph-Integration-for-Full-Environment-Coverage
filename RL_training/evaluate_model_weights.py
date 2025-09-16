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

    if agent_config["name"] == "reinforce":
        if navigation_config["use_transformer"]:
            models_path = Path("RL_training") / "runs" / "model_weights" / "REINFORCE_Agent_Transformer"
        else:
            models_path = Path("RL_training") / "runs" / "model_weights" / "REINFORCE_Agent_LSTM"
    elif agent_config["name"] == "a2c":
        if navigation_config["use_transformer"]:
            models_path = Path("RL_training") / "runs" / "model_weights" / "A2C_Agent_Transformer"
        else:
            models_path = Path("RL_training") / "runs" / "model_weights" / "A2C_Agent_LSTM"
    else:
        raise Exception("Unknown agent")

    # Setup environment
    env = PrecomputedThorEnv(rho=env_config["rho"], max_actions=agent_config["num_steps"])

    # Load agent from encoder & policy weights
    if agent_config["name"] == "reinforce":
        agent = ReinforceAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    elif agent_config["name"] == "a2c":
        agent = A2CAgent(env=env, navigation_config=navigation_config, agent_config=agent_config)
    else:
        raise Exception("Unknown agent")

    for file in models_path.glob("*.pth"):
        print(f"[INFO] Loading model weights from: {file}")
        agent.load_weights(model_path=file, device=device)
        agent.scene_numbers = agent.all_scene_numbers[agent.num_scenes : agent.num_scenes + 3]  # Use next 3 scenes for evaluation
        # RL training runner
        runner = RLEvalRunner(env=env, agent=agent, device=device)
        runner.run()
        env.close()


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
    from components.environments.precomputed_thor_env import PrecomputedThorEnv
    from RL_training.runner.rl_eval_runner import RLEvalRunner
    from components.utils.utility_functions import read_config, set_seeds

    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str, help="Path to the configuration files.")
    args = parser.parse_args()

    # Iterate over each configuration file in args.conf_path
    conf_files = Path(args.conf_path).rglob("*.json")
    conf = read_config(list(conf_files)[0])
    set_seeds(conf["seed"])
    main(conf)
