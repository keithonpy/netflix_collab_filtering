import numpy as np
import kmeans
import common
import em_for_mat
import em

X = np.loadtxt("datasets/toy_data.txt")

# ###############  K-means  ###########################

# for k in range(1, 5):
# 	min_cost = float("inf")
# 	best_seed = 0
# 	for seed in range(5):
# 		mixture, post = common.init(X, k, seed)
# 		mixture, post, cost = kmeans.run(X, mixture, post)
# 		# find the best seed that give the minimum cost
# 		if cost < min_cost:
# 			min_cost = cost
# 			best_seed = seed

# 	mixture, post = common.init(X, k, best_seed)
# 	mixture, post, cost = kmeans.run(X, mixture, post)

# 	title = f"K-means | K = {k}| seed = {best_seed} | cost = {cost}"
# 	print(title)
# 	common.plot(X, mixture, post, title)

# ################ EM algo ############################

# for k in range(1, 5):
# 	max_ll = -float("inf")
# 	best_seed = 0
# 	for seed in range(5):
# 		mixture, post = common.init(X, k, seed)
# 		mixture, post, ll = em.run(X, mixture, post)
# 		# find the best seed that give maximum log-likelihood
# 		if ll > max_ll:
# 			max_ll = ll
# 			best_seed = seed

# 	mixture, post = common.init(X, k, best_seed)
# 	mixture, post, ll = em.run(X, mixture, post)

# 	title = f"EM algo | K = {k}| seed = {best_seed} | ll = {ll}"
# 	print(title)
# 	common.plot(X, mixture, post, title)

############## BIC select the best K ################

max_bic = None
for K in range(1, 5):
    max_ll = None
    best_seed = None
    for seed in range(0, 5):
        mixture, post = common.init(X, K, seed)
        mixture, post, ll = em.run(X, mixture, post)
        if max_ll is None or ll > max_ll:
            max_ll = ll
            best_seed = seed

    mixture, post = common.init(X, K, best_seed)
    mixture, post, ll = em.run(X, mixture, post)
    bic = common.bic(X, mixture, ll)
    if max_bic is None or bic > max_bic:
        max_bic = bic
    title = f"EM algo | K= {K} |seed= {best_seed}, ll= {ll}, bic= {bic}"
    print(title)
    common.plot(X, mixture, post, title)