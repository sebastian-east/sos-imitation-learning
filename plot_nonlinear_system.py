import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import seaborn
import sympy as sp
from sympy import expand

from config.systems import NonlinearSystem


if __name__ == "__main__":
    seaborn.set()
    seaborn.set_style("ticks")
    seaborn.set_context("paper")
    rc('text', usetex=True)
    rc('font', **{'family':'serif', 'serif':['Times New Roman]']})

    # Create plot of imitation loss during training

    fig, ax = plt.subplots(2, 1, figsize=[3.2, 4])

    class Colors:
        blue = (0, 160/255, 255/255)
        orange = (255/255, 160/255, 0/255)
        red = (255/255, 80/255, 0/255)
        green = (0/255, 200/255, 0/255)
        grey = (100/255, 100/255, 100/255)
        lightgrey = (150/255, 150/255, 150/255)

    red = Colors.red
    blue = Colors.blue

    files = os.listdir('./data/results')
    for file in files:

        data = np.load('./data/results/' + file)
        _, system, algorithm, seed, N = file[:-4].split('_')

        if system == 'system':
            if N == '10':
             c = 'black'
            elif N == '100':
                c = red
            elif N == '1000':
                c = blue

            if algorithm == 'admm':
                #plt.semilogy(data['res_pri'])
                ax[0].semilogy(data['obj'][:11], color=c)
            else:
                ax[1].semilogy(data['obj'], color=c)

    line0, = ax[0].plot(0, 0, 'black')
    line1, = ax[0].plot(0, 0, color=red)
    line2, = ax[0].plot(0, 0, color=blue)
    line0.set_label('N=10')
    line1.set_label('N=100')
    line2.set_label('N=1000')

    ax[0].grid()
    ax[0].set_ylim([1E-1, 1E6])
    ax[0].set_xlim([-.5, 10])
    ax[0].set_title('ADMM', loc='left')
    ax[0].set_ylabel('Imitaiton Loss')
    ax[0].set_xlabel('Iteration')
    ax[1].set_ylim([1E-1, 1E6])
    ax[1].set_xlim([-50, 1000])
    ax[1].grid()
    ax[1].set_title('Projected Gradient Descent', loc='left')
    ax[1].set_ylabel('Imitaiton Loss')
    ax[1].set_xlabel('Iteration')
    fig.legend(handles=[line0, line1, line2], ncol=3, bbox_to_anchor=[1.05, 0.08],
               frameon=False)
    seaborn.despine(trim="true")
    fig.tight_layout(pad=0, h_pad=1)
    fig.subplots_adjust(bottom=0.19)
    plt.savefig('./data/figures/training1.pgf', pad_inches=0)
    plt.show()

    # Generate the contour and streamline plot for a learned system

    def vector(x, y, system, states):
        xdot = system.evalf(subs={states[0]:x, states[1]:y})
        #dynamics = sp.lambdify(states, np.squeeze(system), "numpy")
        return float(xdot[0]), float(xdot[1])

    def dynamics(x, t, system, states):
        dyn = sp.lambdify(states, np.squeeze(system), "numpy")
        return np.array(dyn(*x))

    def lyap(x, y, function, states):
        return np.log10(float(function.evalf(subs={states[0]:x, states[1]:y})))

    file = './data/results/nonlinear_system_admm_0_1000.npz'
    data= np.load(file, allow_pickle=True)
    t = NonlinearSystem()
    F = data['F']
    P = data['P']
    Pinv = np.linalg.inv(P.astype('float'))
    learned_system = (t.A + t.B @ F @ Pinv) @ t.Z
    learned_lyapunov = expand((t.Z.T @ Pinv @ t.Z)[0])

    fig = plt.figure(figsize=[2.2, 1.8])
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Vector = np.vectorize(vector, excluded=[2, 3])
    Lyapunov = np.vectorize(lyap, excluded=[2, 3])
    vx, vy = Vector(X, Y, learned_system, t.states)
    Z = Lyapunov(X, Y, learned_lyapunov, t.states)
    plt.contourf(X, Y, Z)
    plt.streamplot(X, Y, vx, vy, color='k', density=0.55)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.grid()
    fig.tight_layout(pad=0)
    plt.savefig('./data/figures/Lyapunov1.pgf', pad_inches=0)
    plt.show()