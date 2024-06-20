from gymnasium.envs.registration import register

register(
    id="gym_xarm/XarmLift-v0",
    entry_point="gym_xarm.tasks:Lift",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

register(
    id="gym_xarm/XarmPush-v0",
    entry_point="gym_xarm.tasks:Push",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)
