import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from random_agent import RandomAgent
from always_on_agent import AlwaysOnAgent
from no_op_agent import NoOpAgent
from dqn_agent import DQNAgent
from handcrafted_features import *
from sarsa_agent_v2 import SarsaAgentV2
from reinforce_agent import ReinforceAgent




env = Environment_v1()

agents_to_compare = [
    #  Description,                             Instance
    ( "Random agent",                           RandomAgent(env)                                                                                       ),
    ( "Always On agent",                        AlwaysOnAgent(env)                                                                                     ),
    ( "No-op agent",                            NoOpAgent(env)                                                                                         ),
    ( "DQN agent\n(hand-crafted features)",       DQNAgent(env, Q_FeatureVector(HandCraftedFeatureVector(env)), name="DQN_handcrafted")       ),
    ( "DQN agent\n(minimal state space)",         DQNAgent(env, Q_FeatureVector(MinimalFeatureVector(env))    , name="DQN_minimal")           ),
    ( "SARSA agent\n(hand-crafted features)",     SarsaAgentV2(env, HandCraftedFeatureVector(env)                        , name="SARSA_handcrafted")     ),
    ( "SARSA agent\n(minimal state space)",       SarsaAgentV2(env, MinimalFeatureVector(env)                            , name="SARSA_minimal")         ),
    ( "REINFORCE agent\n(hand-crafted features)", ReinforceAgent(env, HandCraftedFeatureVector(env)                      , name="REINFORCE_handcrafted") ),
    ( "REINFORCE agent\n(minimal state space)",   ReinforceAgent(env, MinimalFeatureVector(env)                          , name="REINFORCE_minimal")     )
]





num_runs = 1000
results_filename = "returns_comparison.pkl"
results = []


for desc, agent in agents_to_compare:
    returns = []
    for _ in range(num_runs):
        rewards = agent.run_episode()
        returns.append( sum(rewards) )
    mean = np.mean(returns)
    std = np.std(returns)
    results.append( (desc, mean, std) )
    print(desc)
    print("\tmean: %.2f\tstd: %.2f" % (mean, std))
    
    
with open("./data/" + results_filename, mode='wb') as results_file:
    pickle.dump(results, results_file)



with open("./data/" + results_filename, mode='rb') as results_file:
    results = pickle.load(results_file)

descs = [d for d, _, _ in results]
means = [m for _, m, _ in results]
stds  = [s for _, _, s in results]

fig, ax = plt.subplots()
btm = min(means) - 10
ax.bar(descs, means-btm, yerr=stds, bottom=btm, alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel("Mean Return (with std. dev.)")
ax.set_title("Average Performance of Agent Algorithms")
ax.yaxis.grid(True)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.savefig("./data/returns_comparison_figure.png")
plt.show()

