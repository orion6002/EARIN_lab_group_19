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
        self.length = length

    def decode(self, lower_bound, upper_bound, aoi):
        seg = self.genes[lower_bound:upper_bound]
        val = int("".join(map(str, seg)), 2)
        max_val = 2**len(seg) - 1
        return min_max_norm(val, 0, max_val, aoi[0], aoi[1])

    def mutation(self, probability):
        if np.random.rand() < probability:
            gene = np.random.randint(0, self.length)
            self.genes[gene] = 1 - self.genes[gene]

    def crossover(self, other):
        c_point = np.random.randint(1, self.length)
        offspring_1 = np.concatenate((self.genes[:c_point], other.genes[c_point:]))
        offspring_2 = np.concatenate((other.genes[:c_point], self.genes[c_point:]))
        return Chromosome(self.length, offspring_1), Chromosome(self.length, offspring_2)


def objective_function(*args):
    return 0.5 * sum(x**4 - 16*x**2 + 5*x for x in args)


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
        return self.objective_function(*self.decode_full_chromosome(chromosome))

    def tournament_selection(self):
        parents = []
        for i in range(self.population_size):
            candidates = np.random.choice(self.population, self.tournament_size)
            most_promissing = min(candidates, key=self.eval_objective_func)
            parents.append(Chromosome(most_promissing.length, most_promissing.genes.copy()))
        return parents

    def reproduce(self, parents):
        if np.random.rand() > self.crossover_probability:
            for parent in parents:
                parent.mutation(self.mutation_probability)
            return parents
        children = []
        for p1, p2 in zip(parents[:self.population_size//2], parents[self.population_size//2:]):
            c1, c2 = p1.crossover(p2)
            c1.mutation(self.mutation_probability)
            c2.mutation(self.mutation_probability)
            children.append(c1)
            children.append(c2)
        self.population = children
        return children
    
    # helper mathod to simplify computing eval_obj_fnc and run 
    def decode_full_chromosome(self, chromosome):
        args = []
        for i in range(self.obj_func_num_args):
            lo = i * self.bits_per_arg
            hi = lo + self.bits_per_arg
            args.append(chromosome.decode(lo, hi, self.aoi[i]))
        return args
             


    def plot_func(self, trace_args):
        # build the contour grid
        x = np.linspace(self.aoi[0][0], self.aoi[0][1], 300)
        y = np.linspace(self.aoi[1][0], self.aoi[1][1], 300)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)
        plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar()
        n = len(trace_args)
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, n))
        x1s = [a[0] for a in trace_args]
        x2s = [a[1] for a in trace_args]
        plt.scatter(x1s, x2s, c=colors, s=20, zorder=3)
        plt.title("Objective function contour with best individual trace")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    def run(self):
        plot_args = []
        plot_vals = []
        for i in range(self.num_steps):
            best = min(self.population, key=self.eval_objective_func)
            plot_args.append(self.decode_full_chromosome(best))
            plot_vals.append(self.eval_objective_func(best))
            parents = self.tournament_selection()
            self.population = self.reproduce(parents)
        return min(self.population, key=self.eval_objective_func), plot_args

# TODO: fill in the parameters for your group and uncomment to run
# I keep this in order to remember the initial given parameters
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


# I found on the internet that:
# For the Styblinski-Tang function, the known global minimum is your ground truth:
# Global minimum value: ≈ −78.3323
# Located at: (x1, x2) ≈ (−2.9035, −2.9035)

def run_experiment(config_name, params):
    results = []
    last_trace = None
    
    print(f"Experiment: {config_name}")
    for i in range(10):
        ga = GeneticAlgorithm(
            chromosome_length=32,
            obj_func_num_args=2,
            objective_function=objective_function,
            aoi=[[-5, 5], [-5, 5]],
            num_steps=100,
            **params
        )
        best_chrom, trace_args = ga.run()
        results.append(ga.eval_objective_func(best_chrom))
        last_trace = trace_args
        
    mean_val = np.mean(results)
    std_val = np.std(results)
    
    print(f"--> Local minimum found: {mean_val:.4f}\n")
    print(f"--> Standard deviation: {std_val:.4f}\n")
    ga.plot_func(last_trace)

test_configs = {
    "High Mutation probability": {"mutation_probability": 0.8, "tournament_size": 3, "population_size": 50, "crossover_probability": 0.8},
    "Large Tournament": {"mutation_probability": 0.02, "tournament_size": 10, "population_size": 50, "crossover_probability": 0.8},
    "Small Population": {"mutation_probability": 0.05, "tournament_size": 2, "population_size": 20, "crossover_probability": 0.7},
    "Best Theorical Baseline": {"mutation_probability": 0.05, "tournament_size": 3, "population_size": 100, "crossover_probability": 0.9}
}

print("Theorical minimum: −78.3323")
for name, params in test_configs.items():
    run_experiment(name, params)