# https://gist.github.com/btbytes/79877
# https://nathanrooy.github.io/posts/2016-08-17/simple-particle-swarm-optimization-with-python/

"""
PSO
"""

import random
import numpy as np


class PsoHyperParam:
    def __init__(self, inertia_, c1_, c2_, population_size_):
        self.inertia = inertia_
        self.c1 = c1_
        self.c2 = c2_
        self.population_size = population_size_


class Particle:
    def __init__(self, dim_, loss_fn_):
        self.dim = dim_
        self.param = self.init_param()
        self.p_loss = loss_fn_(self.param)
        self.l_param = self.init_param()    # l: local best (individual best)
        self.l_loss = loss_fn_(self.l_param)
        self.direction = np.array([0 for _ in range(self.dim)])

    def init_param(self, min_=0, max_=100):
        _params = [None for _ in range(self.dim)]
        for _idx in range(self.dim):
            _params[_idx] = random.randint(min_, max_)
        return np.array(_params)

    def update_param(self, hp, g_param):
        # g_param: parameter of global best particle
        # hp: PsoHyperParam class
        r1 = random.random()
        r2 = random.random()
        social_term = hp.c2 * r2 * (g_param - self.param)
        cognitive_term = hp.c1 * r1 * (self.l_param - self.param)

        self.direction = hp.inertia * self.direction + social_term + cognitive_term
        self.param = self.param + self.direction
        return self.param


def loss_fn(args):
    _sum = 0
    for val in args:
        _sum += abs(val)
    return _sum


if __name__ == "__main__":
    # init
    col_num = 2

    # set hyper parameter
    hp = PsoHyperParam(inertia_=0.1, c1_=2, c2_=2, population_size_=100)

    # make particles
    particles = [None for _ in range(hp.population_size)]
    for idx in range(hp.population_size):
        particles[idx] = Particle(dim_=col_num, loss_fn_=loss_fn)

    # initialize global particle
    g = Particle(dim_=col_num, loss_fn_=loss_fn)
    g_loss = loss_fn(g.param)

    p = Particle(dim_=col_num, loss_fn_=loss_fn)

    count = 0
    while True:
        # optimize particle
        for p in particles:
            # evaluate loss of particle
            p.param = np.array(list(map(int, p.param)))     # to int
            loss = loss_fn(p.param)

            # update 'local' loss & param
            if loss < p.l_loss:
                p.l_loss = loss
                p.l_param = p.param

            # update 'global' loss & param
            if loss < g_loss:
                g_loss = loss
                g.param = p.param

            print("\ncount:   ", count)
            print("g loss:  ", g_loss)
            print("p loss:  ", loss)
            print("p param: ", p.param, "\n")

            count += 1

            # update particle
            p.update_param(hp=hp, g_param=g.param)

            # stop condition
            if g_loss < 0.1:
                break

        # stop condition
        if g_loss < 0.1:
            break

    print("\ncount:   ", count)
    print("g loss:  ", g_loss)
    print("l loss:  ", p.l_loss)
    print("p loss:  ", loss)
    print("p param: ", p.param, "\n")
