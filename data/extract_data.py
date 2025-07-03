import os
import matplotlib.pyplot as plt
import numpy as np


def mean_deviation(data):
    mean = np.mean(data)  
    deviations = [abs(x - mean) for x in data]  # Calculate absolute deviations
    return np.mean(deviations)  # Calculate the mean of absolute deviations



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
    directory = os.getcwd() 
    for filename in os.listdir(directory):
        if filename.startswith('argon.lj.'):
            filepath = os.path.join(directory, filename)
            time_step, x, y, z, vx, vy, vz, fx, fy, fz, etot = extract_data(filepath)
            total_energy = etotal(etot)
            results.append((time_step, total_energy))

    time_steps = [result[0] for result in results]
    energies = [result[1] for result in results]
    print(mean_deviation(energies))

     # Plotting the results
    plt.figure(figsize=(20, 3))
    plt.plot(time_steps, energies, marker='*', linestyle='None', color='b')  
    plt.xlabel('Time Step')
    plt.ylabel('Total Energy')
    plt.ylim(-18.4,-18)
    plt.grid()
    plt.show()
