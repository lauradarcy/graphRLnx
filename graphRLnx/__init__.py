from gym.envs.registration import register

register(
    id='graphRL-v0',
    entry_point='graphRLnX.envs:graphRLnX',
    kwargs={'network_size': 10, 'input_nodes': 3}
)