import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("datasets/toy_data.txt")

###############  K-means  ###################


for k in range(1, 5):
	min_cost = float("inf")
	best_seed = 0
	for seed in range(5):
		mixture, post = common.init(X, k, seed)
		mixture, post, cost = kmeans.run(X, mixture, post)
		# find the best seed that give the minimum cost
		if cost < min_cost:
			min_cost = cost
			best_seed = seed

	mixture, post = common.init(X, k, best_seed)
	mixture, post, cost = kmeans.run(X, mixture, post)

	title = f"K-means | K = {k}| seed = {seed} | cost = {cost}"
	print(title)
	common.plot(X, mixture, post, title)
