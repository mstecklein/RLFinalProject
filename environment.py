from typing import Tuple
import numpy as np
import gym
from gym.utils import seeding
import pyglet
from gym.envs.classic_control import rendering
from gym.wrappers.monitoring.video_recorder import VideoRecorder


"""
::: Environment-v0 :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Problem setup:
 - Sensor positions are always the same, and have option to place them symmetrically
 - Source positions and transition probabilities are random
 - 3-coverage is possible for all grid squares (and therefore all sources
   can be found)
 - Problem terminates after all sources have been found, or some number of runs

Action space:
 - # actions = 2 * number sensors + 1
 - first half of actions is to turn on sensors
 - second half of actions is to turn off sensors
 - last action is a no-op

Reward:
 - Reward_t = R * (# srcs located at time t) - C * (# snsrs on at t)
   
POMDP state space:
 - Comprehensive state space is kept internally
 - Only part of the state space is observable by the agent, making this env a POMDP
 
 State Space Component  |     Type     |     Shape      | Observable |       Description
 ----------------------------------------------------------------------------------------------------------------------------------------
 SOURCE_LOCATIONS       |      []      |  num_sources   |     no     | List of (r,c) locations of the sources in the field.
 ----------------------------------------------------------------------------------------------------------------------------------------
 SOURCE_STATUSES        |  np.ndarray  | (num_sources,) |     no     | Contains 1's for locations where sources are on, else 0's.
 ----------------------------------------------------------------------------------------------------------------------------------------
 SOURCE_TURNON_PROBS    |  np.ndarray  | (num_sources,) |     no     | Transition probabilities for sources to turn on that are off.
 ----------------------------------------------------------------------------------------------------------------------------------------
 SOURCE_TURNOFF_PROBS   |  np.ndarray  | (num_sources,) |     no     | Transition probabilities for sources to turn off that are on.
 ----------------------------------------------------------------------------------------------------------------------------------------
 SENSOR_STATUSES        |  np.ndarray  | (num_sensors,) |     yes    | Contains 1's for locations where sensors are on, else 0's.
 ----------------------------------------------------------------------------------------------------------------------------------------
 LOCATED_SOURCES        |  np.ndarray  | (field_shape,) |     yes    | Map of field, with 1's marking locations of sources that have
                        |              |                |            | been located, else 0's.
 ----------------------------------------------------------------------------------------------------------------------------------------
 SENSOR_COVERAGES       |  np.ndarray  | (num_sensors,  |  optional  | Matrix (np.ndarray) of size (num_sensors x (field_shape)),
                        |              |   field_shape) |            | containing 1's where each sensor covers, else 0's.
 ----------------------------------------------------------------------------------------------------------------------------------------
 SENSOR_OBSERVATIONS    |     [{}]     |       --       |  optional* | List of sets. Each set represents an "observation" that occurred
                        |              |                |            | in the previous timestep, and contains sensor numbers of the
                        |              |                |            | sensors which observed the same source.
 ----------------------------------------------------------------------------------------------------------------------------------------
 * this component is returned in the "info" section of the step(.) function

"""



SOURCE_LOCATIONS     = "sourcelocations"
SOURCE_STATUSES      = "sourcestatuses"
SOURCE_TURNON_PROBS  = "sourceturnonprobs"
SOURCE_TURNOFF_PROBS = "sourceturnoffprobs"
SENSOR_STATUSES      = "sensorstatuses"
LOCATED_SOURCES      = "locatedsources"
SENSOR_COVERAGES     = "sensorcoverages"
SENSOR_OBSERVATIONS  = "sensorobservations"


GRID_SQR_DEFAULT_COLOR = (1.,1.,1.) # white
GRID_LINE_COLOR = (.2,.2,.2) # dark gray
SOURCE_COLOR = (1.,0.,0.) # red
SENSOR_COLOR_ON = (0.,0.,1.) # bright blue
SENSOR_COLOR_OFF = (0.,0.,.7) # dull blue
SENSOR_1_COV_COLOR = (.8,.8,1.) # light blue
SENSOR_3_COV_COLOR = (.6,.6,1.) # darker light blue


class Environment_v0(gym.Env):
    
    
    metadata = { 'render.modes': ['human', 'rgb_array'] }
    
    
    def __init__(self, field_shape: Tuple[int, int] = (10,10),
                 num_sensors: int = 20,
                 num_sources: int = 10,
                 sensor_radius: int = 3,
                 source_reward: float = 10.0,
                 sensor_cost: float = 0.1,
                 screen_width: int = 600,
                 max_allowed_steps: int = 1000,
                 place_sensors_symmetric: bool = False,
                 debug: bool = False):
        super().__init__()
        
        self._field_shape = field_shape
        self._num_sensors = num_sensors
        self._num_sources = num_sources
        self._sensor_radius = sensor_radius
        self._reward_per_source = source_reward
        self._cost_per_sensor = sensor_cost
        self._screen_width = screen_width
        self._max_allowed_steps = max_allowed_steps
        self._place_sensors_symmetric = place_sensors_symmetric
        self._debug = debug
        self._viewer= None
        self._prev_located_sources = np.zeros(self._field_shape)
        self._action_count = 0
        self._last_action_desc = ""
        self._last_reward_desc = ""
        
        self._reset_hidden_state()
        self.observation_space = gym.spaces.Dict({
                SENSOR_STATUSES : gym.spaces.MultiBinary(self._hidden_state[SENSOR_STATUSES].size),
                LOCATED_SOURCES : gym.spaces.MultiBinary(self._hidden_state[LOCATED_SOURCES].size),
                SENSOR_COVERAGES: gym.spaces.MultiBinary( (num_sensors,) + field_shape)
        })
        self._num_actions = 2*num_sensors + 1 # turn each sensor on or off, no-op
        self.action_space = gym.spaces.Discrete(self._num_actions)
        
        
    def _reset_hidden_state(self):
        self._hidden_state = {
                SOURCE_LOCATIONS:    [],                           # not observable
                SOURCE_STATUSES:     np.zeros(self._num_sources),  # not observable
                SOURCE_TURNON_PROBS: None,                         # not observable
                SOURCE_TURNOFF_PROBS:None,                         # not observable
                SENSOR_STATUSES:     np.zeros(self._num_sensors),  # observable
                LOCATED_SOURCES:     np.zeros(self._field_shape),  # observable
                SENSOR_COVERAGES:    [],                           # optionally observable (return in "info")
                SENSOR_OBSERVATIONS: []                            # optionally observable (return in "info")
            }
        while not self._ensure_sufficient_3_coverage():
            self._init_sensors()
        self._init_sources()
        
        
    def _ensure_sufficient_3_coverage(self):
        # Check sensor coverages to ensure enough grid squares are covered by at least 3 sensors
        snsr_coverages = self._hidden_state[SENSOR_COVERAGES]
        if len(snsr_coverages) == 0:
            return False
        assert len(snsr_coverages) == self._num_sensors
        cov_cnt_map = np.zeros(self._field_shape)
        for sensor_num in range(self._num_sensors):
            cov_cnt_map += self._hidden_state[SENSOR_COVERAGES][sensor_num]
        for sensor_loc in self._sensor_locs:
            cov_cnt_map[sensor_loc] = 0
        num_avail_spots = np.sum(cov_cnt_map >= 3)
        return num_avail_spots >= self._num_sources
        
        
    def _init_sensors(self):
        if self._place_sensors_symmetric:
            self._init_sensors_symmetric()
        else:
            self._init_sensors_random_consistent()
    
    
    def _init_sensors_symmetric(self):
        # Symmetric, hand-placed
        assert self._field_shape == (10,10), "Symmetric sensor placement only works for fields of shape (10,10)"
        assert self._sensor_radius >= 3, "Symmetric sensor placement only works for sensor_radius>=3"
        pass # initialize sensors symmetrically
        self._sensor_locs = [
            (0,0), (1,1), (1,3), (3,1), (3,3),
            (0,9), (1,6), (1,8), (3,6), (3,8),
            (9,0), (6,1), (8,1), (6,3), (8,3),
            (9,9), (6,6), (6,8), (8,6), (8,8)
            ]
        assert self._num_sensors == len(self._sensor_locs), "Symmetric sensor placement only works for num_sensors=20"
        snsr_coverages = []
        for loc in self._sensor_locs:
            # Create coverage map: each entry is a matrix the same size of the
            # field with 1's where the sensor covers
            coverage = self._create_coverage(loc)
            snsr_coverages.append(coverage)
        self._hidden_state[SENSOR_COVERAGES] = snsr_coverages
        
    
    def _init_sensors_random_consistent(self):
        # Asymmetric but consistent sensor placement:
        np_random, seed = seeding.np_random(123) # ensures same sensor locations
        snsr_coverages = []
        self._sensor_locs = []
        for _ in range(self._num_sensors):
            # Find an unused location in the grid
            loc = self._get_random_location(np_random)
            while (loc in self._sensor_locs) or \
                  (loc in self._hidden_state[SOURCE_LOCATIONS]):
                loc = self._get_random_location(np_random)
            self._sensor_locs.append(loc)
            # Create coverage map: each entry is a matrix the same size of the
            # field with 1's where the sensor covers
            coverage = self._create_coverage(loc)
            snsr_coverages.append(coverage)
        self._hidden_state[SENSOR_COVERAGES] = snsr_coverages
        
    
    def _init_sources(self):
        # create transition probabilities
        self._hidden_state[SOURCE_TURNON_PROBS]  = np.random.uniform(low=1./15., high=1./5., size=self._num_sources)
        self._hidden_state[SOURCE_TURNOFF_PROBS] = np.random.uniform(low=1./ 5., high=1./1., size=self._num_sources)
        # create map of coverage, to ensure 3-coverage where sources are placed
        cov_cnt_map = np.zeros(self._field_shape)
        for sensor_num in range(self._num_sensors):
            cov_cnt_map += self._hidden_state[SENSOR_COVERAGES][sensor_num]
        # find locations
        src_locs = []
        for _ in range(self._num_sources):
            # Find an unused location in the grid that has at least 3-coverage
            loc = self._get_random_location()
            while (loc in src_locs) or (loc in self._sensor_locs) or cov_cnt_map[loc] < 3:
                loc = self._get_random_location()
            src_locs.append(loc)
        self._hidden_state[SOURCE_LOCATIONS] = src_locs
    
    
    def _create_coverage(self, location):
        coverage = np.zeros(self._field_shape)
        for r in range(self._field_shape[0]):
            for c in range(self._field_shape[1]):
                dist = np.sqrt((r - location[0])**2 + (c - location[1])**2)
                if dist <= self._sensor_radius:
                    coverage[r,c] = 1
        return coverage
    
    
    def _get_random_location(self, np_random_generator=None):
        if np_random_generator is None:
            r = np.random.randint(self._field_shape[0])
            c = np.random.randint(self._field_shape[1])
        else:
            r = np_random_generator.randint(self._field_shape[0])
            c = np_random_generator.randint(self._field_shape[1])
        return (r,c)
    
    
    def _get_observable_state(self):
        return {
            SENSOR_STATUSES : self._hidden_state[SENSOR_STATUSES],
            LOCATED_SOURCES : self._hidden_state[LOCATED_SOURCES],
            SENSOR_COVERAGES: self._hidden_state[SENSOR_COVERAGES]
        }
    
    
    def _get_info_observable_state(self):
        return {
            SENSOR_OBSERVATIONS : self._hidden_state[SENSOR_OBSERVATIONS]
        }
        
    
    def _create_coverage_count_map(self):
        map = np.zeros(self._field_shape)
        for sensor_num in range(self._num_sensors):
            if self._hidden_state[SENSOR_STATUSES][sensor_num] == 1: # is on
                map += self._hidden_state[SENSOR_COVERAGES][sensor_num]
        return map
    
    
    def step(self, action):
        # Execute one time step within the environment
        assert(action >= 0)
        assert(action < self._num_actions)
        self._action_count += 1
        # Update sensor statuses
        if action < self._num_sensors: # turn on
            sensor_num = action
            self._hidden_state[SENSOR_STATUSES][sensor_num] = 1
            if self._debug:
                self._last_action_desc = "%d: Turn ON sensor #%d" % (self._action_count, sensor_num)
                print(self._last_action_desc)
        elif action < 2*self._num_sensors: # turn off
            sensor_num = action - self._num_sensors
            self._hidden_state[SENSOR_STATUSES][sensor_num] = 0
            if self._debug:
                self._last_action_desc = "%d: Turn OFF sensor #%d" % (self._action_count, sensor_num)
                print(self._last_action_desc)
        else: # no-op
            if self._debug:
                self._last_action_desc = "%d: No-op" % self._action_count
                print(self._last_action_desc)
        # Update source statuses
        for source_num in range(self._num_sources):
            if self._hidden_state[SOURCE_STATUSES][source_num] == 1: # is on
                if np.random.rand() < self._hidden_state[SOURCE_TURNOFF_PROBS][source_num]:
                    self._hidden_state[SOURCE_STATUSES][source_num] = 0 # turn off
            else: # is off
                if np.random.rand() < self._hidden_state[SOURCE_TURNON_PROBS][source_num]:
                    self._hidden_state[SOURCE_STATUSES][source_num] = 1 # turn on
        # Update sensor observations and located sources
        self._hidden_state[SENSOR_OBSERVATIONS] = []
        num_sources_located = 0
        coverage_cnt_map = self._create_coverage_count_map()
        for source_num in range(self._num_sources):
            src_loc = self._hidden_state[SOURCE_LOCATIONS][source_num]
            # skip sources that have already been discovered or are off
            if self._hidden_state[LOCATED_SOURCES][src_loc] == 1 or \
                self._hidden_state[SOURCE_STATUSES][source_num] == 0:
                continue
            # update located sources
            src_loc_coverage_cnt = coverage_cnt_map[src_loc]
            if src_loc_coverage_cnt >= 3: # located source
                self._hidden_state[LOCATED_SOURCES][src_loc] = 1
                num_sources_located += 1
            # update sensor observations (only for sources that haven't been located, for now)
            observed_set = set()
            for sensor_num in range(self._num_sensors):
                if self._hidden_state[SENSOR_COVERAGES][sensor_num][src_loc] == 1 and \
                    self._hidden_state[SENSOR_STATUSES][sensor_num]: # sensor covers source and is on
                    observed_set.add(sensor_num)
            if len(observed_set) > 0:
                self._hidden_state[SENSOR_OBSERVATIONS].append(observed_set)
        # Return observable state, reward, done, info
        num_sensors_on = np.sum(self._hidden_state[SENSOR_STATUSES])
        reward = self._reward_per_source * num_sources_located - self._cost_per_sensor * num_sensors_on
        num_sources_found = np.sum(self._hidden_state[LOCATED_SOURCES])
        done = (num_sources_found == self._num_sources) or (self._num_actions > self._max_allowed_steps)
        info = self._get_info_observable_state()
        if self._debug:
            self._last_reward_desc = "Reward = R * # srcs located - C * # snsrs on = %.2f * %d - %.2f * %d = %.4f" \
                                    % (self._reward_per_source,num_sources_located,self._cost_per_sensor,num_sensors_on,reward)
            print("\t" + self._last_reward_desc)
        return self._get_observable_state(), reward, done, info
        
        
    def reset(self):
        # Reset the state of the environment to an initial state
        self._reset_hidden_state()
        self._action_count = 0
        return self._get_observable_state()
        
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        screen_width = self._screen_width
        screen_height = screen_width * self._field_shape[0] / self._field_shape[1]
        grid_height = screen_height / self._field_shape[0]
        grid_width  = screen_width  / self._field_shape[1]
        # Initialization of rendering
        if self._viewer is None:
                self._viewer = rendering.Viewer(screen_width, screen_height)
                # grid squares
                self._grid_squares = []
                for row in range(self._field_shape[0]):
                    grid_squares_row = []
                    for col in range(self._field_shape[1]):
                        l,r,t,b = 0, grid_width, 0, grid_height
                        grid_square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                        grid_square.set_color(*GRID_SQR_DEFAULT_COLOR)
                        screen_loc = (row*grid_height, col*grid_width)
                        grid_square.add_attr(rendering.Transform(translation=screen_loc))
                        grid_squares_row.append(grid_square)
                        self._viewer.add_geom(grid_square)
                    self._grid_squares.append(grid_squares_row)
                # grid lines
                for r in range(1, self._field_shape[0]):
                    line = rendering.Line((0.,r*grid_height), (screen_width,r*grid_height))
                    line.set_color(*GRID_LINE_COLOR)
                    self._viewer.add_geom(line)
                for c in range(1, self._field_shape[1]):
                    line = rendering.Line((c*grid_width,0.), (c*grid_width,screen_width))
                    line.set_color(*GRID_LINE_COLOR)
                    self._viewer.add_geom(line)
                # sensors
                self._sensor_geoms = []
                for loc in self._sensor_locs:
                    dot = rendering.make_circle(radius=min(grid_height,grid_width)/4)
                    dot.set_color(*SENSOR_COLOR_OFF)
                    screen_loc = ((loc[0]+.5)*grid_height, (loc[1]+.5)*grid_width)
                    dot.add_attr(rendering.Transform(translation=screen_loc))
                    self._sensor_geoms.append(dot)
                    self._viewer.add_geom(dot)
        # Update grid square colors
        coverage_cnt_map = self._create_coverage_count_map()
        for row in range(self._field_shape[0]):
            for col in range(self._field_shape[1]):
                if coverage_cnt_map[row,col] >= 3:
                    self._grid_squares[row][col].set_color(*SENSOR_3_COV_COLOR)
                elif coverage_cnt_map[row,col] >=1:
                    self._grid_squares[row][col].set_color(*SENSOR_1_COV_COLOR)
                else: # no coverage
                    self._grid_squares[row][col].set_color(*GRID_SQR_DEFAULT_COLOR)
        # Update sensors
        for snsr_status, geom in zip(self._hidden_state[SENSOR_STATUSES], self._sensor_geoms):
            if snsr_status == 1: # on
                geom.set_color(*SENSOR_COLOR_ON)
            else: # off
                geom.set_color(*SENSOR_COLOR_OFF)
        # Add newly discovered sources
        for row in range(self._field_shape[0]):
            for col in range(self._field_shape[1]):
                if self._hidden_state[LOCATED_SOURCES][row,col] == 1 and \
                   self._prev_located_sources[row,col] == 0:
                    dot = rendering.make_circle(radius=min(grid_height,grid_width)/4, filled=True)
                    dot.set_color(*SOURCE_COLOR)
                    screen_loc = ((row+.5)*grid_height, (col+.5)*grid_width)
                    dot.add_attr(rendering.Transform(translation=screen_loc))
                    self._viewer.add_geom(dot)
                    self._prev_located_sources[row,col] = 1
        # Debug renderings
        if self._debug:
            # action description
            self._viewer.add_onetime(self._make_text(self._last_action_desc,
                                                     x=0, y=screen_height,
                                                     font_size=12, anchor_y='top'))
            # reward description
            self._viewer.add_onetime(self._make_text(self._last_reward_desc,
                                                     x=0, y=0,
                                                     font_size=8))
            # coverage count map
            for row in range(self._field_shape[0]):
                for col in range(self._field_shape[1]):
                    text = self._make_text(str(int(coverage_cnt_map[row,col])),
                                           x=row*grid_height, y=col*grid_width, font_size=8, alpha=.5)
                    self._viewer.add_onetime(text)
            # sources
            for loc, status in zip(self._hidden_state[SOURCE_LOCATIONS], self._hidden_state[SOURCE_STATUSES]):
                dot = rendering.make_circle(radius=min(grid_height,grid_width)/8)
                if status == 1: # on
                    dot.set_color(*SOURCE_COLOR)
                else:
                    dot.set_color(.5, 0., 0.)
                screen_loc = ((loc[0]+.5)*grid_height, (loc[1]+.5)*grid_width)
                dot.add_attr(rendering.Transform(translation=screen_loc))
                self._viewer.add_onetime(dot)
        # Return rendering
        return self._viewer.render(return_rgb_array = mode=='rgb_array')
    
    
    def _make_text(self, text, x=0, y=0, font_size=36, \
                   anchor_x='left', anchor_y='bottom', color=(0.,0.,0.), alpha=1.):
        """    anchor_x: left, center, right
               anchor_y: top, center, baseline, bottom
        """
        class TextGeom:
            def __init__(self, label:pyglet.text.Label):
                self.label=label
            def render(self):
                self.label.draw()
        color255 = (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(alpha*255))
        label = pyglet.text.Label(text, font_size=font_size, x=x, y=y, \
                                  anchor_x=anchor_x, anchor_y=anchor_y, color=color255)
        return TextGeom(label)
    
    
    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    print("Running test code from environment.py")
    
    seed = 987
    np.random.seed(seed)
    env = Environment_v0(debug=True, place_sensors_symmetric=True)
    video_recorder = VideoRecorder(env, path="/tmp/gym_env_test_seed%d.mp4"%seed)
    env.reset()
    env.render()
    video_recorder.capture_frame()
    total_reward = 0.
    for i in range(30):
        a = np.random.randint(env.action_space.n)
        _, R, _, _ = env.step(a)
        total_reward += R
        env.render()
        video_recorder.capture_frame()
    video_recorder.close()
    env.close()
    print("Total reward: ", total_reward)
