# graph tool reinforcement learning with networkX

practice making an openAI gym environment for a tabular graph solver

basic code for environment:

## `__init__`:

```python
    def __init__(self, network_size=10, input_nodes=3):
        self.network_size = network_size
        self.input_nodes = input_nodes
        self.graph = Graph()
        self.graph.set_fast_edge_removal(True)
        self.graph.add_vertex(self.network_size)

        self.action_space = spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))
        self.observation_space = spaces.MultiDiscrete(np.full((self.network_size, self.network_size), 2))
        
        self.time_step = 0
        self.observation = adjacency(self.graph).toarray()
        
        self.reset()
```

#### `network_size` and `input_nodes`:

Network size is the total number of services, input nodes are where the "source data" is coming from - all further connections either have to come from an input node, or from a node that has previously received information that came from an input node (recursively).

#### `action_space`:

 should be the list of possible vertex to vertex edges. Note that this should NOT account for the DAG compatibility - we want the action space to be the same for all states, so instead should simply be a tuple that contains all node indices in the graph, i.e. a possible action could be `(0,3)` - make a directed connection from node 0 to node 3. the entire action space should then be `spaces.Tuple((spaces.Discrete(self.network_size), spaces.Discrete(self.network_size)))` - a tuple that allows for an integer value from 0 up to the final node index.

#### `observation_space`:

should allow for possible valid state spaces. The state space for this tabular function will simply be the adjacency matrix of the network - and we want a network with only one edge going from one node to another, so we can make the observation space a matrix the size of the adjacency matrix (a square matrix with length of the number of nodes), filled with the value 2. This means the observation space will only be valid for matrices where all values are either the integers 0 or 1.

## `step(self, action)`:

```python
        def step(self, action):
        assert self.action_space.contains(action)
        valid_source_nodes = [index for index, in_degree in
                              enumerate(self.graph.get_in_degrees(self.graph.get_vertices())) if
                              (in_degree > 0 or index < self.input_nodes)]
        if action[0] not in valid_source_nodes:
            raise ValueError('this action does not have a valid from node')
        new_edge = self.graph.add_edge(action[0], action[1])
        if not is_DAG(self.graph):
            self.graph.remove_edge(new_edge)
            raise ValueError('this action violates the DAG property')
        self.observation = adjacency(self.graph).toarray()
        if not self.observation_space.contains(self.observation):
            self.graph.remove_edge(new_edge)
            self.observation = adjacency(self.graph).toarray()
            raise ValueError('this action makes a duplicate edge')

        return self.observation, reward, done, {"time_step": self.time_step}
```

#### `raise ValueError()`:

these are used to ensure the action given is actually a valid action. If they aren't, the method breaks. When using this environment, you will need to wrap the `step` method in a `try` clause.





