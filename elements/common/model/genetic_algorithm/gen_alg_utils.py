import random
import numpy as np
import matplotlib.pyplot as plt
import operator


class City:
    """
     Object that represents one city
    """

    def __init__(self, x, y, name):
        """
        :param x: coordinate of the city in the x axis
        :param y: coordinate of the city in the y axis
        :name : name of the city
        """
        self.x = x
        self.y = y
        self.name = name

    def distance(self, city):
        """
        Compute the distance between the two cities

        param city:  city to compute the distance to
        """
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness_Function:
    """"
    Object to compute the fitness of a given solution (how good a given route is)
    """

    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        """
        Compute total route distance
        """
        if self.distance == 0:
            path_distance = 0
            # compute the distance from one city to the other and stores in path_distance
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    # from the last city back to the first one
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def route_fitness(self):
        """
        Compute the fitness of the route (to use it as an optimization problem)
        """
        # 1/total distance = the larger the distance, the smaller the fitness
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


def create_random_problem(total_cities):
    """
    Creates random cities in a grid of 200x200

    :params total_cities: total cities in the grid
    :return: list with all the cities created
    """
    city_list = list()
    for i in range(0, total_cities):
        city_list.append(City(x=int(random.random() * 200), y=int(random.random() * 200), name="city" + str(i)))
    return city_list


def create_initial_population(pop_size: int, city_list: []):
    """"
    Create the initial population by random (random order of visiting the cities)

    :param pop_size: size of the population
    :param city_list: list of the class City, with all the cities that must be visited
    :return: list with the whole population
    """
    population = []

    for i in range(0, pop_size):
        population.append((random.sample(city_list, len(city_list))))
    return population


def rank_best_routes(population: list):
    """"
    Ranks the population based on the Fitness Function

    :param population: list of the population
    :return: list with the sorted fitness of each individual in the population
    """

    fitness_results = {}
    for i in range(0,len(population)):
        fitness_results[i] = Fitness_Function(population[i]).route_fitness()
    fitness_results = sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)
    return fitness_results


def breed(parent1, parent2):
    """"
        Cross-over function for breeding the parents and generating the descendents for the new population

        :param parent1: one individual of the population (order of cities to be visited)
        :param parent2: another individual of the population (order of cities to be visited)
        :return: a mix of the solution provided by both parents, a child
    """

    child_p1 = []

    #interval used to copy from the parent 1, the rest will be filled by parent 2
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        child_p1.append(parent1[i])

    # a city cannot be visited twice, so we make sure it does not happen
    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p1 + child_p2
    return child

def mutate(individual, mutation_rate):
    """"
       Add mutation to ta single individual by swapping the order of two cities

       :param individual: one individual of the population (one order of visiting the cities)
       :param mutation_rate: the chance of a mutation occurrying
       :return: the mutated (or not) individual
    """

    for swapped in range(len(individual)):
        # everytime a random mutation occurs, cities are swapped
        if (random.random() < mutation_rate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def breed_population_elitism(matingpool, elite_size):
    """"
        Applies the breed function to the whole population

        :param matingpool: the population selected for mating
        :param elite_size: size of the elite population
        :return: a list with the new population composed by all the children
    """
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))

    # The elite is added without changes, maybe it is already the best solution
    for i in range(0, elite_size):
        children.append(matingpool[i])

    # now the top of the matingpool breeds with the bottom, generating new solutions
    ##hide
    for i in range(0, length):
        #breed randomly switches two cities in a path
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    ##show
    return children

def mating_pool(population, selection_results):
    """"
    Build the mating pool by converting indexes from the selection function to the actual population

    :param population: list with the current population
    :param selection_results: list of the index of each individual of the population selected for mating
    :return: list with the new population to be used for mating and generating the new population
    """
    matingpool = []
    for i in range(0, len(selection_results)):
        index = selection_results[i]
        matingpool.append(population[index])
    return matingpool


def plot_best_route(fit_best,best_route,name):
    cities_values = best_route
    x = list()
    y = list()
    for value in cities_values:
        x.append(value.x)
        y.append(value.y)

    plt.scatter(x[0], y[0], color='Green')
    #plt.annotate("START", (x[0], y[0]))
    plt.scatter(x[-1], y[-1], color='Red')
    #plt.annotate("END", (x[-1], y[-1]))
    plt.plot(x, y)
    for i in (range(0,len(cities_values))):
        plt.annotate(cities_values[i].name, [cities_values[i].x,cities_values[i].y])
    if isinstance(fit_best,list):
        plt.title("%s Distance: %f.2"% (name, 1 / fit_best[0][1]))
    else:
        plt.title("%s Distance: %f.2"% (name, 1 / fit_best))
    plt.show()