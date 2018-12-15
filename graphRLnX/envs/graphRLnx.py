import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import networkx as nx


class graphRLnx(gym.Env):
    """
    will have fixed action space, but not all actions are valid within each state
    step function should have a function that tests if the chosen action is valid
    the observation returned will be the graph_tool graph, but the state will just be
    the adjacency matrix (? maybe, currently have obs space as the matrix)
    maybe step function just alters the given graph
    """
    metadata = {'render.modes': ['human', 'graph', 'interactive']}

    def __init__(self, network_size=10, input_nodes=3):
        self.network_size = network_size
        self.input_nodes = input_nodes
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.network_size))

        self.action_space = spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))
        self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        self.time_step = 0
        self.observation = nx.to_numpy_matrix(self.graph).astype(int)
        self.seed_value = self.seed()
        self.true_graph = self.create_true_graph()
        self.reset()

    def create_true_graph(self):
        final_workflow = nx.DiGraph()
        final_workflow.add_nodes_from(range(self.network_size))
        i = 0
        while nx.ancestors(final_workflow,self.network_size-1) != set(range(self.network_size)):
            i += 1
            if i > 10000:
                raise RuntimeError('generating graph took too long')
            valid_source_nodes = [index for index, in_degree in
                                  final_workflow.in_degree() if
                                  ((in_degree > 0 or index < self.input_nodes) and index < (self.network_size - 1))]
            valid_to_nodes = [index for index in range(self.input_nodes, self.network_size)]
            new_edge = [(self.np_random.choice(valid_source_nodes), self.np_random.choice(valid_to_nodes))]
            final_workflow.add_edges_from(new_edge)
            if not nx.algorithms.dag.is_directed_acyclic_graph(final_workflow):
                final_workflow.remove_edges_from(new_edge)
            observation = nx.to_numpy_matrix(final_workflow).astype(int)
            if not self.observation_space.contains(observation):
                final_workflow.remove_edges_from(new_edge)
        return final_workflow

    def render(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.graph
        elif mode == 'interactive':
            interactive_window(self.graph)
        elif mode == 'human':
            filename = "./renders/render" + str(self.time_step) + ".png"
            graph_draw(self.graph, vertex_text=self.graph.vertex_index, vertex_font_size=18,
                       output_size=(1000, 1000), output=filename)

    def render_truth(self, mode='human'):
        if mode == 'graph':
            # return graphtools graph object
            return self.true_graph
        elif mode == 'interactive':
            interactive_window(self.true_graph)
        elif mode == 'human':
            filename = "./renders/TrueGraphSeed" + str(self.seed_value) + ".png"
            graph_draw(self.true_graph, vertex_text=self.true_graph.vertex_index, vertex_font_size=18,
                       output_size=(1000, 1000), output=filename)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        done = 0
        reward = 0
        assert self.action_space.contains(action)
        valid_source_nodes = [index for index, in_degree in
                              enumerate(self.graph.get_in_degrees(self.graph.get_vertices())) if
                              ((in_degree > 0 or index < self.input_nodes) and index < (self.network_size - 1))]
        if action[0] not in valid_source_nodes:
            raise ValueError('this action does not have a valid from node')
        new_edge = self.graph.add_edge(action[0], action[1])
        if not is_DAG(self.graph):
            self.graph.remove_edge(new_edge)
            raise ValueError('this action violates the DAG property')
        self.observation = adjacency(self.graph).toarray().astype(int)
        if not self.observation_space.contains(self.observation):
            print(self.observation)
            self.graph.remove_edge(new_edge)
            self.observation = adjacency(self.graph).toarray().astype(int)
            raise ValueError('this action makes a duplicate edge')
        if isomorphism(self.graph, self.true_graph):
            reward = 1
            done = 1
        self.time_step += 1
        return self.observation, reward, done, {"time_step": self.time_step}

    def reset(self):
        self.graph.clear_edges()
        self.time_step = 0
        self.observation = adjacency(self.graph).toarray()
        return self.observation
