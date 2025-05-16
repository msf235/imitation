import numpy as np
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout as rollout_module

from imitation.algorithms import bco

from stable_baselines3.common.evaluation import evaluate_policy

if __name__ == "__main__":
    # 1. Load expert policy
    rng = np.random.default_rng(0)
    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals:seals/CartPole-v0",
        venv=env,
    )
    print("Expert policy ready")

    # 2. Roll out expert to collect transitions (obs-only)
    rollouts_expert = rollout_module.rollout(
        expert,
        env,
        rollout_module.make_sample_until(min_episodes=50),
        rng=rng,
    )
    expert_transitions = rollout_module.flatten_trajectories(rollouts_expert)
    print(f"Collected {len(expert_transitions.obs)} expert transitions")

    # 3. Generate random IDM demonstrations
    idm_transitions = bco.generate_random_demonstrations(
        env=env, num_episodes=50, rng=rng
    )
    print(f"Collected {len(idm_transitions.obs)} random IDM transitions")

    # 4. Instantiate BCOfromObservation
    bco_trainer = bco.BCO(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        demonstrations=expert_transitions,
        idm_demonstrations=idm_transitions,
        continuous=False,
    )
    print("BCO agent instantiated")

    # reward_before_training, _ = evaluate_policy(bco_trainer.policy_net, env, 10)
    # print(f"Reward before training: {reward_before_training}")
    bco_trainer.train_idm(n_epochs=10)

    # action_loss_before_training, _ = evaluate_policy(bco_trainer.policy_net, env, 10)
    # print(f"Reward before training: {reward_before_training}")
    #
    bco_trainer.train(n_epochs=1)

    reward_after_training, _ = evaluate_policy(bco_trainer.policy, env, 10)
    print(f"Reward after training: {reward_after_training}")
