# -*- coding: utf-8 -*-

import os
import sys
import numpy as np


class HMM:
    """
    Hidden Markov Model. Viterbi for decoding, Baum-welch for learning.

    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    pi: numpy.ndarray
        Initial state probablity vector

    Common Variables
    ----------------
    obs_seq : list of int
        list of observations (represented as ints corresponding to output
        indexes in B) in order of appearance
    T : int
        number of observations in an observation sequence
    N : int
        number of states
    """

    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi

    def _forward(self, obs_seq):
        "前向计算观测序列的概率"
        N = self.A.shape[0]
        T = len(obs_seq)

        F_prop = np.zeros((N, T))
        F_prop[:, 0] = self.pi * self.B[:, obs_seq[0]]  # t=1的序列概率
        for t in range(1, T):
            for n in range(N):
                # t step时状态为n的概率（已知观测序列）
                # F_prop[:, t-1]: t-1 step所有的状态概率向量
                # self.A[:, n]：所有状态转移到n状态的概率向量
                # self.B[n, obs_seq[t]]：n状态对应观测为obs_seq[t]的概率
                F_prop[n, t] = np.dot(
                    F_prop[:, t-1], self.A[:, n]) * self.B[n, obs_seq[t]]

        return F_prop

    def _backward(self, obs_seq):
        N = self.A.shape[0]
        T = len(obs_seq)

        B_prop = np.zeros((N, T))
        B_prop[:, -1:] = 1
        for t in reversed(range(T - 1)):
            for n in range(N):
                # B_prop[:, t+1]: t+1 step所有的状态概率向量
                # self.A[n, :]: 从n状态转移到其他所有状态的概率向量
                # self.B[:, obs_seq[t+1]]：t+1步所有状态对应观测到obs_seq[t+1]的概率向量
                B_prop[n, t] = np.sum(
                    B_prop[:, t+1] * self.A[n, :] * self.B[:, obs_seq[t+1]])

        return B_prop

    def observation_prob_forward(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        return np.sum(self._forward(obs_seq)[:, -1])

    def observation_prob_backward(self, obs_seq):
        """ P( entire observation sequence | A, B, pi ) """
        beta_slice = self._backward(obs_seq)[..., 0]
        b_start = self.B[..., obs_seq[0]]
        return np.sum(self.pi * beta_slice * b_start)

    def viterbi(self, obs_seq):
        """
        Returns
        -------
        V : numpy.ndarray
            V [s][t] = Maximum probability of an observation sequence ending
                       at time 't' with final state 's'
        prev : numpy.ndarray
            Contains a pointer to the previous state at t-1 that maximizes
            V[state][t]
        """
        N = self.A.shape[0]
        T = len(obs_seq)
        prev = np.zeros((T - 1, N), dtype=int)

        # DP matrix containing max likelihood of state at a given time
        V = np.zeros((N, T))
        V[:, 0] = self.pi * self.B[:, obs_seq[0]]

        for t in range(1, T):
            for n in range(N):
                # seq_probs：t时刻为n状态时，遍历t-1时刻所有可能状态得到的概率向量
                # prev[t-1, n]：t时刻为n状态时，t-1时刻最优的状态
                # V[n, t]：t时刻为n状态时，最大的概率值
                seq_probs = V[:, t-1] * self.A[:, n] * self.B[n, obs_seq[t]]
                prev[t-1, n] = np.argmax(seq_probs)
                V[n, t] = np.max(seq_probs)

        return V, prev

    def decoding(self, obs_seq):
        """
        Returns
        -------
        V[last_state, -1] : float
            Probability of the optimal state path
        path : list(int)
            Optimal state path for the observation sequence
        """
        V, prev = self.viterbi(obs_seq)

        last_state = np.argmax(V[:, -1])
        path = list(self.build_viterbi_path(prev, last_state))

        return V[last_state, -1], reversed(path)

    def build_viterbi_path(self, prev, last_state):
        """Returns a state path ending in last_state in reverse order."""
        T = len(prev)
        yield(last_state)
        for i in range(T - 1, -1, -1):
            yield(prev[i, last_state])
            last_state = prev[i, last_state]

    def baum_welch_train(self, observations, criterion=0.05):
        """已知观测序列，求解A、B. 前向后向算法：E step 计算t时刻状态为qi的期望和
        状态从qi转移到qj的概率期望。M step 计算A、B的值。直到收敛为止。"""
        N = self.A.shape[0]
        T = len(observations)
        done = False
        while not done:
            # E step:

            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = self._forward(observations)
            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = self._backward(observations)

            # t时刻状态从qi转移到qj的概率期望矩阵
            xi = np.zeros((N, N, T-1))
            for t in range(T - 1):
                # 公式分母
                # np.dot(alpha[:, t].T, self.A): 得到一个行向量，每个位置表示前向计算到t时刻为该位置对应状态的概率
                denom = np.dot(np.dot(alpha[:, t].T, self.A) * self.B[:, observations[t+1]].T,
                                beta[:, t+1])
                for i in range(N):
                    # 公式分子
                    # qj隐含在向量的列维度
                    numer = alpha[i, t] * self.A[i, :] * \
                            self.B[:, observations[t+1]].T * \
                            beta[:, t+1].T
                    xi[i, :, t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)：在xi的基础上消去t+1的状态
            gamma = np.squeeze(np.sum(xi, axis=1))
            # Need final gamma element for new B
            prod = (alpha[:, T - 1] *
                    beta[:, T - 1]).reshape((-1, 1))
            # append one more to gamma：因为xi中只计算到T-1时刻，T时刻的观测期望还没有计算
            gamma = np.hstack((gamma, prod / np.sum(prod)))  # column wise

            # M step:

            newpi = gamma[:, 0]
            newA = np.sum(xi, 2) / \
                np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))

            newB = np.copy(self.B)
            observe_types = self.B.shape[1]
            sumgamma = np.sum(gamma, axis=1)
            # t时刻为qj状态且观测为observations[t]的期望概率
            for num in range(observe_types):
                mask = observations == num
                newB[:, num] = np.sum(gamma[:, mask], axis=1) / sumgamma

            if np.max(abs(self.pi - newpi)) < criterion and \
                    np.max(abs(self.A - newA)) < criterion and \
                    np.max(abs(self.B - newB)) < criterion:
                done = True

            self.A[:], self.B[:], self.pi[:] = newA, newB, newpi

    def simulate(self, T):
        "测试数据生成"
        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0], :])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t - 1], :])
            observations[t] = draw_from(self.B[states[t], :])
        return observations, states


if __name__ == "__main__":
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 3)
    pi = np.random.rand(4)
    A = A / np.sum(A, axis=1, keepdims=True)
    B = B / np.sum(B, axis=1, keepdims=True)
    pi = pi / np.sum(pi, axis=0, keepdims=True)
    model = HMM(A, B, pi)

    print("===initialized===")
    print(A)
    print(pi)

    print("===simulate data===")
    T = 10
    observations, states = model.simulate(T)
    print("观测序列", observations)

    print("===learning===")
    model.baum_welch_train(observations)
    print(model.A)
    print(model.pi)

    print("===decoding===")
    prob, path = model.decoding(observations)
    print(list(path))