import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm

def extract_data(filename):
    with open(filename,  encoding='ISO-8859-1') as file:
        data = file.readlines()

    time_step = None
    x    =   []
    y    =   []
    z    =   []
    vx   =   []
    vy   =   []
    vz   =   []
    fx   =   []
    fy   =   []
    fz   =   []
    etot =   []
    atom_data_started = False

    for line in data:
        if line.startswith("ITEM: TIMESTEP"):
            time_step = int(data[data.index(line) + 1].strip())
        if line.startswith("ITEM: ATOMS"):
            atom_data_started = True
            continue
        if atom_data_started:
            if line.strip() == "":
                break
            components = list(map(float, line.split()[2:]))
            x.append(components[0])
            y.append(components[1])
            z.append(components[2])
            vx.append(components[3])
            vy.append(components[4])
            vz.append(components[5])            
            fx.append(components[6])
            fy.append(components[7])
            fz.append(components[8])             
            etot.append(components[9])
    
    return time_step, x, y, z, vx, vy, vz, fx, fy, fz, etot

def etotal(e_vec):
    return sum(e_vec)

if __name__ == "__main__":
    results = []
    all_velocities = []
    directory = os.getcwd() 
    for filename in os.listdir(directory):
        if filename.startswith('argon.lj.'):
            filepath = os.path.join(directory, filename)
            time_step, x, y, z, vx, vy, vz, fx, fy, fz, etot = extract_data(filepath)
            if etotal(etot) == 0:
                continue
            v = np.sqrt(np.array(vx)**2 + np.array(vy)**2 + np.array(vz)**2)
            total_energy = etotal(etot)
            results.append((time_step, total_energy))
            all_velocities.append(v)


    time_steps = [result[0] for result in results]
    energies = [result[1] for result in results]
    
    mean_e = np.mean(energies)
    sigma_e = np.std(energies)
    print(f'mu = {mean_e:.6f}')
    print(f'sigma = {sigma_e:.6f}')
    print(f'Min. energy = {np.min(energies)}')
    print(f'Max. energy = {np.max(energies)}')

    plt.figure(figsize=(10, 3))
    plt.plot(time_steps, energies, marker='*', linestyle='None', color='b')  
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy')
    plt.ylim(mean_e - 10*sigma_e, mean_e + 10*sigma_e)

    plt.axhline(y=mean_e, color='r', linestyle='-', label=f'Mean energy (μ) = {mean_e:.6f}')
    plt.axhline(y=mean_e+sigma_e, color='k', linestyle='--', label=f'μ + σ = {mean_e + sigma_e:.6f}')
    plt.axhline(y=mean_e-sigma_e, color='k', linestyle='--', label=f'μ - σ = {mean_e - sigma_e:.6f}')

    ####Histogram of Energy
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(energies, bins=40, density=True, edgecolor='black')
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    mu, sigma = norm.fit(energies)

    # Plot the PDF
    x = np.linspace(min(energies), max(energies), 100)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', lw=2)

    plt.axvline(mean_e, color='r', linestyle='dashed', linewidth=2, label=f'Mean (μ) = {mean_e:.6f}')
    plt.axvline(mean_e + sigma, color='g', linestyle='dashed', linewidth=2, label=f'μ + σ = {mean_e + sigma:.6f}')
    plt.axvline(mean_e - sigma, color='g', linestyle='dashed', linewidth=2, label=f'μ - σ = {mean_e - sigma:.6f}')

    plt.xlabel('Total Energy')
    plt.ylabel('Frecuency')
    plt.title('Distribution of Total Energy')
        
    plt.legend()
    plt.grid()
    plt.show()

    ####Histogram of velocities
    hist_matrix = []
    all_v_flat = np.concatenate(all_velocities)
    vmin = np.min(all_v_flat)
    vmax = np.max(all_v_flat)
    step_interval = 100
    for v in all_velocities[::step_interval]:
        counts, _ = np.histogram(v, bins=20, range=(vmin, vmax))
        hist_matrix.append(counts)
    hist_matrix = np.array(hist_matrix)

    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='3d')
    T, B = hist_matrix.shape
    xpos, ypos = np.meshgrid(np.arange(B), np.arange(T))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = 1
    dz = hist_matrix.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

    ax.set_xlabel('Velocity')
    ax.set_ylabel('Time Step / 100')
    ax.set_zlabel('Frequency')
    ax.set_title('Velocities Over Time')

    plt.show()
