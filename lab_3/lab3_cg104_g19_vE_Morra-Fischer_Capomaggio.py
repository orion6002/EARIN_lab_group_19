import numpy as np
import matplotlib.pyplot as plt


def min_max_norm(val, min_val, max_val, new_min, new_max):
    return (val - min_val) * (new_max - new_min) / (max_val - min_val) + new_min


class Chromosome:
    def __init__(self, length, array=None):  # if array is None, initialize with a random binary vector
        if array is None:
            self.genes = np.random.randint(0, 2, length)
        else:
            self.genes = np.array(array)
        self.lenght = self.lenght

    def decode(self, lower_bound, upper_bound, aoi):
        val = int("".joint(map(str, self.genes)), 2)
        max_val = 2**self.lenght - 1
        return min_max_norm(val, 0, max_val, aoi[0], aoi[1])

    def mutation(self, probability):
        if np.random.rand() < probability:
            gene = np.random.randint(0, self.lenght)
            self.genes[gene] = 1 - self.genes[gene]

    def crossover(self, other):
        c_point = np.random.randint(1, self.lenght)
        offspring_1 = np.concatenate((self.genes[:c_point], other.genes[c_point:]))
        offspring_2 = np.concatenate((other.genes[:c_point], self.genes[c_point:]))
        return Chromosome(self.lenght, offspring_1), Chromosome(self.lenght, offspring_2)


# TODO: implement your group's objective function here
def objective_function(*args):
    return 0.5 * (args[0]**4 - 16*(args[0]**2) + 5*args[0] + args[1]**4 - 16*(args[1]**2) + 5*args[1])


class GeneticAlgorithm:
    def __init__(self, chromosome_length, obj_func_num_args, objective_function, aoi,
                 population_size=100, tournament_size=2, mutation_probability=0.05,
                 crossover_probability=0.8, num_steps=50):
        assert chromosome_length % obj_func_num_args == 0, "Number of bits for each argument should be equal"
        self.chromosome_length = chromosome_length
        self.obj_func_num_args = obj_func_num_args
        self.bits_per_arg = int(chromosome_length / obj_func_num_args)
        self.objective_function = objective_function
        self.aoi = aoi
        self.tournament_size = tournament_size
        self.mutation_probability = mutation_probability
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.num_steps = num_steps
        
        self.population = [Chromosome(chromosome_length) for i in range(population_size)]

    def eval_objective_func(self, chromosome):
        args = []
        for i in range (self.obj_func_num_args):
            arg = chromosome.genes[i * self.bits_per_arg : (i+1) * self.bits_per_arg]
            new_chrom = Chromosome(self.bits_per_arg, arg)
            args.append(new_chrom.decode(0, self.bits_per_arg - 1, self.aoi[i]))
        return self.objective_function(*args)

    def tournament_selection(self):
        parents = []
        for i in range(self.population_size):
            candidates = np.random.choice(self.population, self.tournament_size)
            most_promissing = min(candidates, key=self.eval_objective_func)
            parents.append(most_promissing)
        return parents


    def reproduce(self, parents):
        pass

    def plot_func(self, trace):
        plt.plot(trace)
        plt.title("Algorithm convergence")
        plt.xlabel("Gens")
        plt.ylabel("Value of obj function")
        plt.show()

    def run(self):
        points = []
        for i in range(self.num_steps):
            most_promissing = min(self.population, key=self.eval_objective_func)
            points.append(self.eval_objective_func(most_promissing))
            parents = self.tournament_selection()
            self.population = self.reproduce(parents)
        self.plot_func(points)
        return min(self.population, key=self.eval_objective_func)

# TODO: fill in the parameters for your group and uncomment to run
# ga = GeneticAlgorithm(
#     chromosome_length=...,
#     obj_func_num_args=2,
#     objective_function=objective_function,
#     aoi=[...],
#     population_size=...,
#     tournament_size=2,
#     mutation_probability=0.05,
#     crossover_probability=0.8,
#     num_steps=...
# )
# ga.run()