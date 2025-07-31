import sympy

class FitnessFunctions:
    @staticmethod
    def f1(individual_performance):
        fitness = individual_performance.get("best_f1", 0.0)
        return fitness
    
    @staticmethod
    def train_loss(individual_performance):
        fitness = individual_performance.get("train_loss", 0.0)
        return fitness
    
    @staticmethod
    def prime_fitness(branches):
        return sum(item for branch in branches for item in branch if sympy.isprime(item))
    
    @staticmethod
    def closeness_to_global_opt(branches):
        global_optimum = [[19, 18], [420, 120, 8], [1000, 1000, 1000]]
        # Sort each branch in both the input and global optimum
        sorted_branches = sorted(branches, key=lambda x: len(x))
        score = 0
        # Length mismatch penalty
        if len(sorted_branches) != len(global_optimum):
            score -= 10_000
            return score  # Early return for severe mismatch
        # Compare each corresponding branch
        for branch, optimum_branch in zip(sorted_branches, global_optimum):
            # Length mismatch within branch
            if len(branch) != len(optimum_branch):
                score -= 1_000
                continue
            # Calculate element-wise distance (using Manhattan distance)
            for a, b in zip(branch, optimum_branch):
                score -= abs(a - b)  # Negative because lower distance is better
        return score
    
    @staticmethod
    def train_loss_normalize(individual, population):

        losses = [x.fitness.values[0] for x in population]
        highest_loss_val = max(losses)
        lowest_loss_val = min(losses)
        loss = individual.fitness.values[0]

        if highest_loss_val == lowest_loss_val:
            fitness = 1.0
        else:
            fitness = (highest_loss_val - loss) / (highest_loss_val - lowest_loss_val)

        individual.fitness.values = (fitness,)

    @staticmethod
    def no_normalization(individual, population):
        pass

    MINIMIZE_FITNESS = True
    fitness_function = train_loss
    normalization_function = no_normalization