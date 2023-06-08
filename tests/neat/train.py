
import argparse
import time
import flappy_bird_gymnasium
import gymnasium
import numpy as np

from config import Config
from observer import Observer
from population import Population
from utils import RELU,SIGMOID,TANH



env = gymnasium.make("FlappyBird-v0")
def create_fitness_function(reps = 5):
    def fitness_function(network):
        final = 0
        for _ in range(reps):
            obs, _ = env.reset() 
            cummulative_return = 0
            while True:
                action_values = np.exp(np.array(network.predict(obs)))
                action_values = action_values/np.sum(action_values)
                if(action_values[1] > 0.8):
                    action = 1
                else:
                    action = 0
                
                obs, reward, terminated, _, info = env.step(action)
                
                cummulative_return += np.int32(reward)
                if terminated:
                    break
            final += cummulative_return            
        return final/reps + 1
    return fitness_function





def train(generations,
          pop_size, 
          remove_node_mutation_rate, 
          remove_connection_mutation_rate, 
          add_node_mutation_rate, 
          add_connection_muttation_rate, 
          target_species):
    stagnetion_limit = 50
    miu_prop = 0.3
    crossover_rate = 0.8
    crossover_rate_of_reenable = 0.2

    activation_funtions = [RELU, SIGMOID, TANH]
    activation_funtion_mutation_rate = 0.2
    weight_mutation_rate = 0.9
    step_size = 1
    C1 = 1
    C2 = 1
    C3 = 0.5
    delta = 5



    config = Config(env.observation_space.shape[0], env.action_space.n, lamda=pop_size, 
                    stagnetion_limit=stagnetion_limit, miu_prop= miu_prop, crossover_rate=crossover_rate,
                    crossover_rate_of_reenable= crossover_rate_of_reenable, activation_funtions= activation_funtions,
                    activation_funtion_mutation_rate=activation_funtion_mutation_rate, weight_mutation_rate=weight_mutation_rate,
                    step_size = step_size, remove_node_mutation_rate= remove_node_mutation_rate, remove_connection_mutation_rate = remove_connection_mutation_rate,
                    add_node_mutation_rate = add_node_mutation_rate, add_connection_muttation_rate= add_connection_muttation_rate,
                    target_species = target_species, C1=C1, C2=C2, C3=C3, delta = delta)
    observer = Observer()
    pop = Population(config,fitness_function=create_fitness_function(5))
    pop.train(generations,observer)

    best = pop.best

    np.save("scores",observer.fitnesses)


    return best

def play(network, plays):
    for _ in range(plays):
        obs, _ = env.reset() 
        cummulative_return = 0
        while True:
            action_values = np.exp(np.array(network.predict(obs)))
            action_values = action_values/np.sum(action_values)
            if(action_values[1] > 0.8):
                action = 1
            else:
                action = 0
            
            obs, reward, terminated, _, info = env.step(action)
            env.render()
            time.sleep(0.1)
            cummulative_return += np.int32(reward)
            if terminated:
                break
        print(f"Best replay score: {cummulative_return+1}")


        env.close()
        time.sleep(0.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NEAT')
    parser.add_argument('--generations', type=int, default=10, help='Number of generation to run')
    parser.add_argument('--pop_size', type=int, default=300, help='Population size')
    parser.add_argument('--remove_node', type=float, default=0.5, help='Probability to remove a node')
    parser.add_argument('--remove_connection', type=float, default=0.5, help='Probability to remove a connection')
    parser.add_argument('--add_node', type=float, default=0.3, help='Probability to add a node')
    parser.add_argument('--add_connection', type=float, default=0.3, help='Probability to add a connection')
    parser.add_argument('--target_species', type=int, default=20, help='Number of target species')
    parser.add_argument('--plays', type=int, default=3, help='Number of final simulations')
    args = parser.parse_args()

    best = train(args.generations,args.pop_size,args.remove_node,args.remove_connection, args.add_node,args.add_connection, args.target_species)
    play(best.create_network(),args.plays)



