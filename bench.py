# load modules
# ------------
import sys
import platform
import cpuinfo
from timeit import default_timer as timer
import numpy as np
from scipy.integrate import odeint
from scipy.linalg import lu
from scipy import sparse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from drawnow import drawnow, figure
from rich.table import Table
from rich.console import Console

# get optional parameters
# -----------------------
param = dict()
for element in sys.argv[1:]:
    key, value = element.split("=")
    param[key.lstrip("-")] = value

# number of runs
try:
    n = int(param['n'])
except KeyError:
    n = 1
# export
try:
    exp = bool(int(param['export']))
    fout = 'results.txt'
except KeyError:
    exp = False
# import
try:
    imp = bool(int(param['import']))
    fin = 'other-results.txt'
except KeyError:
    imp = False

# set defaults
# ------------
times = []

# BEGIN BENCHMARK
# ---------------

# LU
print(f"Performing LU benchmark ...", end=' ', flush=True)
inner_times = []
for foo in range(n):
    t1 = timer()
    A = np.random.random((3000,3000))
    p, l, u = lu(A)
    t2 = timer()
    inner_times.append(t2 - t1)
    del t1, A, p, l, u, t2
times.append(inner_times)
print("done", flush=True)


# FFT
print(f"Performing FFT benchmark ...", end=' ', flush=True)
inner_times = []
for foo in range(n):
    t = np.linspace(0,4*np.pi, 5000000)
    t1 = timer()
    sp = np.fft.fft(np.sin(t))
    t2 = timer()
    inner_times.append(t2 - t1)
    del t, t1, sp, t2
times.append(inner_times)
print("done", flush=True)

# ODE
# Reference: http://math.colgate.edu/math329/exampleode.py
print(f"Performing ODE benchmark ...", end=' ', flush=True)
inner_times = []
def vanderpol(y,t,mu):
    """ Return the derivative vector for the van der Pol equations."""
    y1= y[0]
    y2= y[1]
    dy1=y2
    dy2=mu*(1-y1**2)*y2-y1
    return [dy1, dy2]

def run_vanderpol(yinit=[2,0], tfinal=20, mu=2):
    """ Example for how to run odeint.

    More info found in the doc_string. In ipython type odeint?
    """
    times = np.linspace(0,tfinal,2000)

    rtol=1e-6
    atol=1e-10

    y = odeint(vanderpol, yinit, times, args= (mu,), rtol=rtol, atol=atol)
    return y,times

for foo in range(n):
    t1 = timer()
    y,t = run_vanderpol([2,0], tfinal=3500, mu=1)
    t2 = timer()
    inner_times.append(t2 - t1)
times.append(inner_times)
print("done", flush=True)

# SPARSE
# Reference: https://stackoverflow.com/questions/21097657/numpy-method-to-do-ndarray-to-vector-mapping-as-in-matlabs-delsq-demo
print(f"Performing SPARSE benchmark ...", end=' ', flush=True)
inner_times = []

def numgrid(n):
    """
    NUMGRID Number the grid points in a two dimensional region.
    G = NUMGRID('R',n) numbers the points on an n-by-n grid in
    an L-shaped domain made from 3/4 of the entire square.
    adapted from C. Moler, 7-16-91, 12-22-93.
    Copyright (c) 1984-94 by The MathWorks, Inc.
    """
    x = np.ones((n,1))*np.linspace(-1,1,n)
    y = np.flipud(x.T)
    G = (x > -1) & (x < 1) & (y > -1) & (y < 1) & ( (x > 0) | (y > 0))
    G = np.where(G,1,0) # boolean to integer
    k = np.where(G)
    G[k] = 1+np.arange(len(k[0]))
    return G

def delsq(G):
    """
    DELSQ  Construct five-point finite difference Laplacian.
    delsq(G) is the sparse form of the two-dimensional,
    5-point discrete negative Laplacian on the grid G.
    adapted from  C. Moler, 7-16-91.
    Copyright (c) 1984-94 by The MathWorks, Inc.
    """
    [m,n] = G.shape
    # Indices of interior points
    G1 = G.flatten()
    p = np.where(G1)[0]
    N = len(p)
    # Connect interior points to themselves with 4's.
    i = G1[p]-1
    j = G1[p]-1
    s = 4*np.ones(p.shape)

    # for k = north, east, south, west
    for k in [-1, m, 1, -m]:
       # Possible neighbors in k-th direction
       Q = G1[p+k]
       # Index of points with interior neighbors
       q = np.where(Q)[0]
       # Connect interior points to neighbors with -1's.
       i = np.concatenate([i, G1[p[q]]-1])
       j = np.concatenate([j,Q[q]-1])
       s = np.concatenate([s,-np.ones(q.shape)])
    # sparse matrix with 5 diagonals
    return sparse.csr_matrix((s, (i,j)),(N,N))


A = delsq(numgrid(200))
b = np.sum(A, axis=1)
b = sparse.csr_matrix(b)

for foo in range(n):
    t1 = timer()
    x = sparse.linalg.spsolve(A, b)
    t2 = timer()
    inner_times.append(t2 - t1)
times.append(inner_times)
print("done", flush=True)

# 2D plot "imshow"
print(f"Performing 2d plot (1) benchmark ...", end=' ', flush=True)
inner_times = []

alpha = np.linspace(0, 2 * np.pi, 100)
beta = np.linspace(0, 2 * np.pi, 100)
a, b = np.meshgrid(alpha, beta)
npt = 10
data = np.ones((npt, 100, 100))
for i in range(npt):
    data[i] = np.sin(np.roll(a, int(np.ceil(100/4/n))*i, axis=1)) * np.cos(np.roll(b, int(np.ceil(100/4/n))*i, axis=0))

for foo in range(n):
    fig, ax = plt.subplots()
    t1 = timer()
    for i in range(len(data)):
        ax.cla()
        ax.imshow(data[i])
        ax.set_title("run {}".format(foo + 1))
        plt.pause(0.0001)
    t2 = timer()
    inner_times.append(t2 - t1)
    plt.close('all')
times.append(inner_times)
print("done", flush=True)

# 2D plot "drawnow"
print(f"Performing 2d plot (2) benchmark ...", end=' ', flush=True)
# Creating equally spaced 100 data in range 0 to 2*pi
inner_times = []

theta = np.linspace(0, 2 * np.pi, 50)
for foo in range(n):
    figure()
    t1 = timer()
    for i in range(len(theta)):
        def heart():
            # Generating x and y data
            x = 16 * ( np.sin(theta[:i+1]) ** 3 )
            y = 13 * np.cos(theta[:i+1]) - 5* np.cos(2*theta[:i+1]) - 2 * np.cos(3*theta[:i+1]) - np.cos(4*theta[:i+1])

            # Plotting
            plt.cla()
            plt.xlim(-17.59, 17.50)
            plt.ylim(-18.42, 13.32)
            plt.plot(x, y)
            plt.title("run {}".format(foo + 1))

        drawnow(heart)
    t2 = timer()
    inner_times.append(t2 - t1)
    plt.close('all')

times.append(inner_times)
print("done", flush=True)

# 3D plot "imshow"
print(f"Performing 3d plot benchmark ...", end=' ', flush=True)
inner_times = []

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)

npt = 10
k = np.linspace(0.1, 1.5, npt)
data = np.ones((npt, 100, 100))
for i, ele in enumerate(k):
    data[i] = Z = np.sin(ele * R)


for foo in range(n):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    t1 = timer()
    for i in range(len(data)):
        ax.cla()
        surf = ax.plot_surface(X, Y, data[i], rstride=1, cstride=1,
                       linewidth=0, antialiased=False, cmap='viridis')
        ax.set_zlim3d(-1, 1)
        ax.set_title("run {}".format(foo + 1))
        plt.pause(0.0001)
    t2 = timer()
    inner_times.append(t2 - t1)
    plt.close('all')

times.append(inner_times)
print("done\n\n", flush=True)


# Compute average and best values
# -------------------------------
times = np.array(times)
t_ave = np.sum(times, axis=1) / n
t_min = np.min(times, axis=1)

# build summary Table
# -------------------
table = Table(title=f"Benchmark - number of run={n}", show_lines=True)
#
table.add_column("Computer type", justify="left")
table.add_column("Python Version", justify="left", no_wrap=True)
table.add_column("LU", justify="left", no_wrap=True)
table.add_column("FFT", justify="left", no_wrap=True)
table.add_column("ODE", justify="left", no_wrap=True)
table.add_column("SPARSE", justify="left", no_wrap=True)
table.add_column("2D plot\n(imshow)", justify="left", no_wrap=True)
table.add_column("2D plot\n(drawnow)", justify="left", no_wrap=True)
table.add_column("3D plot", justify="left", no_wrap=True)
#
table.add_row("This machine (average)", platform.python_version() , *("{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}".format(*t_ave)).split("-"), style="cyan")
if n > 1:
    table.add_row(f"This machine (best values over {n} runs)", platform.python_version() , *("{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}-{:.4f}".format(*t_min)).split("-"), style="green")

# import results if any
if imp:
    with open(fin, 'r') as file:
        for line in file:
            line_split = line.split(";")
            table.add_row(line_split[0].strip(), line_split[1].strip(), *[f"{float(ele):.4f}" for ele in line_split[2:]])

console = Console()
console.print(table)


# export results
# --------------
if exp:
    this_machine = cpuinfo.get_cpu_info()['brand_raw'] + " - " + platform.system()
    python_version = platform.python_version()
    with open(fout, 'w') as file:
        file.write(this_machine + " ; " + python_version)
        for ele in t_min:
            file.write(f" ; {ele:.6e}")
