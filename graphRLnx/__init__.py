from gym.envs.registration import register

register(
    id='graphRL-v0',
    entry_point='graphRLnx.envs:graphRLnx',
    kwargs={'network_size': 10, 'input_nodes': 3}
)