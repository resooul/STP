import numpy as np
import  copy

class STPMPA():

    def __init__(self, epoch=10000, pop_size=100, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def initialize_variables(self):
        self.FADS = 0.2
        self.P = 0.7

    def levy_step(self, beta, size=None):
        num = np.random.gamma(1 + beta) * np.sin(np.pi * beta /2)
        den = np.random.gamma((1+beta)/2) * beta * 2**((beta-1)/2)
        sigma_u = (num/den) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):

        CF = 1
        RL = 0.05 * self.levy_step(1.5, (self.pop_size, self.problem.n_dims))
        RB = np.random.randn(self.pop_size, self.problem.n_dims)

        for idx in range(0, self.pop_size):

            if epoch > 0: # FADs
                if np.random.rand() < self.FADS:
                    u = np.where(np.random.rand(self.problem.n_dims) < self.FADS, 1, 0)
                    pos_new = self.pop[idx][self.ID_POS] + CF * (
                                self.problem.lb + np.random.rand(self.problem.n_dims) * (
                                    self.problem.ub - self.problem.lb)) * u

                else:
                    per1 = np.random.permutation(self.pop_size)
                    per2 = np.random.permutation(self.pop_size)
                    r = np.random.rand()

                    fi = ((r - 0.5) * 2) * RB[idx]

                    pos_new = copy.deepcopy(self.pop[idx][self.ID_POS])
                    pos_new = self.pop[idx][self.ID_POS] + (
                            fi * (self.pop[per1[idx]][self.ID_POS] - self.pop[per2[idx]][self.ID_POS]))

                pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
                target = self.get_target_wrapper(pos_new)

                self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])
                self.g_best = self.get_better_solution(self.pop[idx], self.g_best)

            R = np.random.rand(self.problem.n_dims)
            t = epoch + 1

            if t < self.epoch / 3:  # Phase 1
                step_size = RB[idx] * (self.g_best[self.ID_POS] - RB[idx] * self.pop[idx][self.ID_POS])
                pos_new = self.pop[idx][self.ID_POS] + self.P * R * step_size
            elif self.epoch / 3 < t < 2 * self.epoch / 3:  # Phase 2
                if idx > self.pop_size / 2:
                    step_size = RB[idx] * (RB[idx] * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                    pos_new = self.g_best[self.ID_POS] + self.P * CF * step_size
                else:
                    step_size = RL[idx] * (self.g_best[self.ID_POS] - RL[idx] * self.pop[idx][self.ID_POS])
                    pos_new = self.pop[idx][self.ID_POS] + self.P * R * step_size
            else:  # Phase 3
                step_size = RL[idx] * (RL[idx] * self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS])
                pos_new = self.g_best[self.ID_POS] + self.P * CF * step_size

            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)

            target = self.get_target_wrapper(pos_new)
            self.pop[idx] = self.get_better_solution(self.pop[idx], [pos_new, target])