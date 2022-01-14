from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
#import tensorflow
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import time as tm
import random

from powerprediction.utils.data_reader import basic_argparser, read_matlab
from powerprediction.utils.data_utils import train_val_test_split, get_font_dict
from powerprediction.utils.data_utils import mape, plot_target, plot_map, plot_predictions, shuffle_train_val


def generate_chromosome(map_len, nb_of_genes):
    """
    Generating a chromosome(array) and sorting the values in ascending order.

    :param map_len: total number of points in the wind map
    :param nb_of_genes: number of points that have to be generated
    :return: the generated chromosome representing a set of points randomly selected from the full map
    """

    genes = random.sample(range(map_len), nb_of_genes)
    genes.sort()
    return genes


def compute_chromosome_score(chromosome, x_train_flat, y_train, x_test_flat, y_test, clf_name="Ridge", metric="mse"):
    """
    Score function that for a given train/test set and a selected set of pixels computes the score of a given
    score metric and a given model name.

    :param chromosome: selected set of points from a map
    :param x_train_flat: X training data
    :param y_train:  y training data
    :param x_test_flat: X test data
    :param y_test: y test data
    :param clf_name: the name of the used regression model
    :param metric: evaluation metric i.e. mse, mape
    :return: the score obtained by the selected model using the selected metric
    """

    new_x_train_flat = x_train_flat[:, chromosome]
    new_x_test_flat = x_test_flat[:, chromosome]

    reg = None
    x_val, y_val, x_test = None, None, None
    if clf_name == "Rf":
        reg = RandomForestRegressor(
            n_estimators=100,
            criterion="mse",
            n_jobs=-1,
            verbose=0,
            min_samples_leaf=10,
            random_state=17
        )
    elif clf_name == "Ridge":
        reg = Ridge(alpha=10)
    elif clf_name == "Lgbm":
        reg = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=50, learning_rate=0.07, n_estimators=150, n_jobs=-1,
                                random_state=17)
        split_num = 5
        x_val = new_x_test_flat[:new_x_test_flat.shape[0] // split_num, :]
        y_val = y_test[:y_test.shape[0] // split_num]
        x_test = new_x_test_flat[new_x_test_flat.shape[0] // split_num:, :]
        y_test = y_test[y_test.shape[0] // split_num:]
    else:
        reg = KNeighborsRegressor(n_neighbors=17)

    y_pred = None
    if clf_name == "Lgbm":
        reg.fit(new_x_train_flat, y_train, eval_set=[(x_val, y_val)], eval_metric="l2", early_stopping_rounds=10,
                verbose=False)
        y_pred = reg.predict(x_test, num_iteration=reg.best_iteration_)
    else:
        reg.fit(new_x_train_flat, y_train)
        y_pred = reg.predict(new_x_test_flat)
    print(y_test)
    print(y_pred)

    if metric == "mse":
        #return keras.losses.mean_squared_error(y_test, y_pred).numpy()
        return mean_squared_error(y_test, y_pred)
    return mean_absolute_percentage_error(y_test, y_pred)


def roulette_bin_search(nr, intervals):
    """
    Binary search for choosing the chromosome interval.

    :param nr: the number that we have to search for
    :param intervals: list of numbers representing edges of intervals ex: 0, 0.006, 0.0013 ... 0.0098, 1
                      we have to see in which interval our number fits
    :return: the interval index in which our number fits
    """

    st = 0  # Binary search
    dr = len(intervals) - 1
    mij = (st + dr) // 2
    # print(mij)

    while st < dr:
        if intervals[mij] < nr and nr < intervals[mij + 1]:
            st = mij
            break
        if nr > intervals[mij + 1]:
            st = mij + 1
        if nr < intervals[mij]:
            dr = mij - 1

        mij = (st + dr) // 2

    return st


def tournament_round(players):
    """
    Tournament selection round, meaning that we choose the chromosome with the highest score of the fitness
    function out of the whole list of chromosomes.

    :param players: list of sets of points that participate to the tournament selection round
    :return: the winner chromosome
    """

    players = sorted(players, key=lambda x: x[1], reverse=True)
    print(players)
    winner, _ = players[0]

    return winner


def crossover(parent1, parent2, map_len):
    """
    Crossover of two chromosomes.
    We will create two new child chromosomes out of two parent chromosomes by combining 1 half of each parent
    chromosome with 1 half of the other one. The genes of every chromosome are pixels from the map.

    Example:
    Parent 1 => (1, 3, 7, 8, 9, 11)                   Child 1 => (_1, _3, 6, _7, 13, 15)
                                         ========>
    Parent 2 => (2, 4, 6, 7, 13, 15)                  Child 2 => (2, 4, 6, _8, _9, _11)

    _x => gene selected from Parent 1

    :param parent1: first set of selected points
    :param parent2: second set of selected points
    :param map_len: the total number a pixels of the map
    :return: two new child chromosomes generated from parent1 and parent2
    """

    print("P1", len(parent1), "P2", len(parent2))
    nr = random.uniform(0, 1)
    child1 = None
    child2 = None

    """ Tried several methods for crossover, the unused ones were commented. 
        They will still be considered in the future development. """

    if nr < 1:
        """
        split_border = len(parent1) // 2 + 1

        child1 = parent1[:split_border] + parent2[split_border:]
        child1_set = set(child1)
        while len(child1_set) != 100:
            pos = random.randrange(map_len)
            while pos in child1_set:
                pos = random.randrange(map_len)
            child1_set.add(pos)
        child1 = list(child1_set)
        child1.sort()

        child2 = parent2[:split_border] + parent1[split_border:]
        child2_set = set(child2)
        while len(child2_set) != 100:
            pos = random.randrange(map_len)
            while pos in child2_set:
                pos = random.randrange(map_len)
            child2_set.add(pos)
        child2 = list(child2_set)
        child2.sort()
        """

        split_border = len(parent1) // 2 + 1

        child1_top = parent1[:split_border]
        child1_top_set = set(child1_top)
        child1_bottom = []
        child1_valid_len = len(parent1) - len(child1_top)
        for elem in parent2[::-1]:
            if elem not in child1_top_set:
                child1_bottom.append(elem)
                if len(child1_bottom) == child1_valid_len:
                    break

        child1 = child1_top + child1_bottom
        child1.sort()

        child2_top = parent2[:split_border]
        child2_top_set = set(child2_top)
        child2_bottom = []
        child2_valid_len = len(parent1) - len(child2_top)
        for elem in parent1[::-1]:
            if elem not in child2_top_set:
                child2_bottom.append(elem)
                if len(child2_bottom) == child2_valid_len:
                    break

        child2 = child2_top + child2_bottom
        child2.sort()
    else:
        split_border = len(parent1) // 2

        child1 = parent1[:split_border] + parent2[split_border:]
        child1_set = set(child1)
        while len(child1_set) != 100:
            pos = random.randrange(map_len)
            while pos in child1_set:
                pos = random.randrange(map_len)
            child1_set.add(pos)
        child1 = list(child1_set)
        child1.sort()

        child2 = parent2[:split_border] + parent1[split_border:]
        child2_set = set(child2)
        while len(child2_set) != 100:
            pos = random.randrange(map_len)
            while pos in child2_set:
                pos = random.randrange(map_len)
            child2_set.add(pos)
        child2 = list(child2_set)
        child2.sort()

        """
        split_border = len(parent1) // 2

        child1_top = parent1[:split_border]
        child1_top_set = set(child1_top)
        child1_bottom = []
        child1_valid_len = len(parent1) - len(child1_top)
        for elem in parent2[::-1]:
            if elem not in child1_top_set:
                child1_bottom.append(elem)
                if len(child1_bottom) == child1_valid_len:
                    break

        child1 = child1_top + child1_bottom
        child1.sort()

        child2_top = parent2[:split_border]
        child2_top_set = set(child2_top)
        child2_bottom = []
        child2_valid_len = len(parent1) - len(child2_top)
        for elem in parent1[::-1]:
            if elem not in child2_top_set:
                child2_bottom.append(elem)
                if len(child2_bottom) == child2_valid_len:
                    break

        child2 = child2_top + child2_bottom
        child2.sort()
        """






        """
        top = [i for i in range(len(parent1)) if i % columns_num <= columns_num // 2]
        bottom = [i for i in range(len(parent1)) if i % columns_num > columns_num // 2]

        child1_top = [parent1[i] for i in top]
        child1_bottom = [parent2[i] for i in bottom]
        child1 = child1_top + child1_bottom
        child1.sort()

        child2_top = [parent2[i] for i in top]
        child2_bottom = [parent1[i] for i in bottom]
        child2 = child2_top + child2_bottom
        child2.sort()
        """

    return child1, child2


def main(args):
    dataset = read_matlab(args.filename, args.dataset)
    load_features = ("wind_speed_100m", "temperature")
    # load_features = ("wind_speed_100m")
    dates = getattr(dataset, "dates").copy()

    x, y = dataset.load_data(features=load_features)
    best_genes = genomics(x, y)
    print(best_genes)


def genomics(x, y, nb_of_genes_list=None, selection_model="Ridge", selection_metric="mse", population_len=300,
             epochs=10, crossover_rate=0.8, mutation_rate=0.005, tournament_size=51, seed_value=17,
             activation_model="Ridge", activation_metric="mse"):
    """ In Depth Genetic Feature Selection Algorithm """

    """ Train test split + shuffle on train and val """
    
    
    # 65% training data, 20% validation data and 15% test data
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x,
                                                                          y,
                                                                          val_percentage=0.20,
                                                                          test_percentage=0.15)

    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    print("x_train_flat shape:", np.shape(x_train_flat))
    print("x_val_flat shape:", np.shape(x_val_flat))
    print("x_test_flat shape:", np.shape(x_test_flat))

    x_train_flat, y_train, x_val_flat, y_val = shuffle_train_val(x_train_flat, y_train, x_val_flat, y_val)
    print("x_train_flat shape after shuffle:", np.shape(x_train_flat))
    print("x_val_flat shape after shuffle:", np.shape(x_val_flat))
    print("x_test_flat shape after shuffle:", np.shape(x_test_flat))

    if len(x.shape) > 2:
        map_len = x.shape[1] * x.shape[2] * x.shape[3]
    else:
        map_len = x.shape[1] * x.shape[2]
    print("Map len:", map_len)

    """ Initialize parameters """
    random.seed(seed_value)
    nb_of_genes_list = nb_of_genes_list if nb_of_genes_list else [50]

    """ Generate initial population """
    genes_scores = []

    for nb_of_genes in nb_of_genes_list:
        chromosomes = []

        for index in range(population_len):
            chromosomes.append(generate_chromosome(map_len, nb_of_genes))

        for i in range(10):
            print(chromosomes[i])

        """ Iterate through generations """
        for epoch in range(epochs):
            print("\n\nEpoch {}:".format(epoch+1))

            """ Selection step: """
            print("Selection:")

            """ Computing chromosome scores """
            print("Computing chromosome scores:")

            scores = []
            total_scores = 0
            for chromosome in chromosomes:
                score = compute_chromosome_score(chromosome, x_train_flat, y_train, x_test_flat, y_test,
                                                 selection_model, selection_metric)

                # Custom Example Of A Hybrid Fitness Function
                # score1 = compute_chromosome_score(chromosome, x_train_flat, y_train, x_val_flat, y_val, "Lgbm", "mse")
                # score2 = compute_chromosome_score(chromosome, x_train_flat, y_train, x_val_flat, y_val, "Ridge","mse")
                # score = (2 * score1 + score2) / 3

                prob = (1/score/100) ** 17
                # prob = (1/score*3000000) ** 17
                print(chromosome)
                print(score, prob)
                print()
                total_scores += prob
                scores.append(prob)

            print("Create score intervals")
            intervals = [0]
            for score in scores:
                prob = score/total_scores
                intervals.append(prob + intervals[-1])

            intervals[-1] = 1

            selected = []
            for i in range(population_len):
                # tournament selection

                players = []
                for step in range(tournament_size):
                    nr = random.uniform(0, 1)
                    player = roulette_bin_search(nr, intervals)
                    score = scores[player]
                    players.append((player, score))

                winner = tournament_round(players)
                selected.append(winner)

            print(selected)
            print(tm.perf_counter())
            """ Crossover """
            print('Crossover')

            prev = 0
            paired = 0
            crossed = []

            for i in range(population_len):
                nr = random.uniform(0, 1)

                if nr < crossover_rate:
                    paired ^= 1

                    if paired == 0:
                        parent1 = chromosomes[selected[prev]]
                        parent2 = chromosomes[selected[i]]

                        child1, child2 = crossover(parent1, parent2, map_len)

                        crossed.append(child1)
                        crossed.append(child2)

                    prev = i

                else:
                    crossed.append(chromosomes[selected[i]])

            if paired == 1:
                crossed.append(chromosomes[selected[prev]])

            print(tm.perf_counter())
            """ Mutation """
            print('Mutation')

            mutated = []

            for i in range(population_len):
                to_mutate_chrom = set(crossed[i])

                for j in range(map_len):
                    nr = random.uniform(0, 1)

                    if nr < mutation_rate:
                        if j in to_mutate_chrom:
                            to_mutate_chrom.remove(j)
                            pos = random.randrange(map_len)
                            while pos in to_mutate_chrom:
                                pos = random.randrange(map_len)
                            to_mutate_chrom.add(pos)
                        else:
                            to_mutate_chrom.add(j)
                            pos = random.randrange(map_len)
                            while pos not in to_mutate_chrom:
                                pos = random.randrange(map_len)
                            to_mutate_chrom.remove(pos)

                to_mutate_chrom_list = list(to_mutate_chrom)
                to_mutate_chrom_list.sort()
                # print(len(to_mutate_chrom_list))
                mutated.append(to_mutate_chrom_list)

            chromosomes = mutated
            print(tm.perf_counter())

        scores = []
        for i, chromosome in enumerate(chromosomes):
            print(chromosome)
            chrom_score = compute_chromosome_score(chromosome, x_train_flat, y_train, x_test_flat, y_test,
                                                 activation_model, activation_metric)
            scores.append((chrom_score, i))

        scores.sort(key=lambda a: a[0])
        best_score = scores[0][0]
        best_chromosome = chromosomes[scores[0][1]]
        print("\n\nBest Chromosome with {} features:".format(nb_of_genes), best_chromosome)
        best_mape_knn = compute_chromosome_score(best_chromosome, x_train_flat, y_train, x_test_flat, y_test, "Knn",
                                                 metric="mape")
        #print("Best Knn MAPE for {} features:".format(nb_of_genes), best_mape_knn)

        best_mape_ridge = compute_chromosome_score(best_chromosome, x_train_flat, y_train, x_test_flat, y_test, "Ridge",
                                                   metric="mape")
        #print("Best Ridge MAPE for {} features:".format(nb_of_genes), best_mape_ridge)

        best_mape_lgbm = compute_chromosome_score(best_chromosome, x_train_flat, y_train, x_test_flat, y_test, "Lgbm",
                                                  metric="mape")
        #print("Best Lgbm MAPE for {} features:".format(nb_of_genes), best_mape_lgbm)

        best_mse_lgbm = compute_chromosome_score(best_chromosome, x_train_flat, y_train, x_test_flat, y_test, "Lgbm",
                                               metric="mse")
        #print("Best Lgbm MSE for {} features:".format(nb_of_genes), best_mse_lgbm)

        #genes_scores.append((best_chromosome, nb_of_genes, best_mape_ridge, best_mape_ridge))
        genes_scores.append((best_chromosome, nb_of_genes, best_score))

    print(genes_scores)

    mape_scores = []
    i = -1
    best_score = 100000
    best_ind = -1
    best_genes = None

    for best_chromosome, _, reg_error in genes_scores:
        i += 1
        mape_scores.append(reg_error)
        if reg_error < best_score:
            best_score = reg_error
            best_ind = i
            best_genes = best_chromosome

    # This part can be commented to stop showing the plot after genetic feature selection
    # font = get_font_dict()
    # plt.title(selection_metric.capitalize() + ' for different numbers of genes (validation set)', fontdict=font)
    # plt.ylabel(selection_metric.capitalize() + ' Score', fontdict=font)
    # plt.xlabel('Number of Points', fontdict=font)
    # plt.plot(nb_of_genes_list, mape_scores, 'co', linestyle='-')
    # plt.show()

    return best_genes


if __name__ == "__main__":
    args = basic_argparser.parse_args()
    main(args)
