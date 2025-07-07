import numpy as np, os, sys, tqdm, pandas as pd



""" LAMMPS Utility Functions -- Myung Chul Kim, July 2025 """
def write_lammps_input_parameterized(filename="create_dataset.in", verbose=False, **kwargs):
    """
    Write LAMMPS input file with customizable parameters
    
    Parameters:
    -----------
    filename : str
        Output filename
    **kwargs : dict
        Parameters to override defaults
    """
    
    # Default parameters
    params = {
        'mass': 39.9,
        'pair_style': 'lj/cut',
        'epsilon': 0.01032,
        'sigma': 3.4,
        'cutoff': 8.5,
        'skin': 2.0,
        'num_replicate': 2,
        'cell_param': 5.46,
        'temp_obj': 10,
        'temp_init': 10,
        'temp_final': 10,
        'press_obj': 137.3,
        'time_step': 0.01078,
        'num_step': 200000,
        'run_step': 200000,
        'dump_freq': 100,
        'log_freq': 200,
        'npt_thermo_freq': 0.02,
        'nvt_thermo_freq': 0.02,
        'npt_baro_freq': 0.2,
        'num_chains': 5,
        'num_mtk': 5,
        'random_seed': 12345
    }
    
    # Update with user-provided parameters
    params.update(kwargs)
    
    lammps_content = f"""# create_dataset.in
# LAMMPS input file for solid Argon MD simulation using LJ potential
# Extracted from create_dataset.py
# Written by Myung Chul Kim, July 2025

# Setup
units metal
boundary p p p
atom_style atomic

# Variables (adjust these as needed)
variable mass equal {params['mass']}
variable epsilon equal {params['epsilon']}  # 0.238 kcal/mol converted to eV
variable sigma equal {params['sigma']}
variable cutoff equal {params['cutoff']}       # 2.5 * sigma
variable skin equal {params['skin']}
variable num_replicate equal {params['num_replicate']}  # approximately (256/4)^(1/3)
variable cell_param equal {params['cell_param']}     # cell/num_replicate for density 0.844
variable temp_obj equal {params['temp_obj']}
variable temp_init equal {params['temp_init']}  
variable temp_final equal {params['temp_final']}
variable press_obj equal {params['press_obj']}
variable time_step equal {params['time_step']}
variable num_step equal {params['num_step']}
variable run_step equal {params['run_step']}
variable dump_freq equal {params['dump_freq']}
variable log_freq equal {params['log_freq']}
variable npt_thermo_freq equal {params['npt_thermo_freq']}
variable nvt_thermo_freq equal {params['nvt_thermo_freq']}
variable npt_baro_freq equal {params['npt_baro_freq']}
variable num_chains equal {params['num_chains']}
variable num_mtk equal {params['num_mtk']}
variable random_seed equal {params['random_seed']}

# Create lattice and atoms
lattice fcc ${{cell_param}}
region box block 0 ${{num_replicate}} 0 ${{num_replicate}} 0 ${{num_replicate}}
create_box 1 box
create_atoms 1 box

# Set mass and initial velocity
mass 1 ${{mass}}
velocity all create ${{temp_obj}} ${{random_seed}} dist gaussian mom yes rot yes

# Define force field (LJ potential)
pair_style {params['pair_style']} ${{cutoff}}
pair_coeff 1 1 ${{epsilon}} ${{sigma}} ${{cutoff}}

# Neighbor settings
neighbor ${{skin}} bin
neigh_modify check yes

# Thermodynamic output
thermo ${{log_freq}}
timestep ${{time_step}}

# Phase 1: NPT ensemble
fix 1 all npt temp ${{temp_obj}} ${{temp_obj}} ${{npt_thermo_freq}} tchain ${{num_chains}} tloop ${{num_mtk}} iso ${{press_obj}} ${{press_obj}} ${{npt_baro_freq}}
run ${{num_step}}
unfix 1

# Phase 2: NVT ensemble (equilibration)
fix 1 all nvt temp ${{temp_obj}} ${{temp_obj}} ${{nvt_thermo_freq}} tchain ${{num_chains}} tloop ${{num_mtk}}
run ${{num_step}}
unfix 1

# Phase 3: NVT ensemble (production run with output)
# Setup dumps
compute pe_atom all pe/atom
compute ke_atom all ke/atom
dump 1 all custom ${{dump_freq}} data/dumps/dump.lj.* id type x y z vx vy vz fx fy fz c_pe_atom
dump 2 all custom ${{dump_freq}} data/dumps/dump.lj id type xu yu zu vx vy vz c_pe_atom

# Setup energy output
thermo ${{log_freq}}
reset_timestep 0
thermo_style custom step pe
variable sstep equal step
variable ppt equal pe
fix extra all print ${{dump_freq}} "${{sstep}} ${{ppt}}" file data/dumps/etotal.txt screen no

# Final NVT run with temperature ramp
fix 1 all nvt temp ${{temp_init}} ${{temp_final}} ${{nvt_thermo_freq}} tchain ${{num_chains}} tloop ${{num_mtk}}
run ${{run_step}}
"""

    with open(filename, 'w') as f:
        f.write(lammps_content)
    
    if verbose: 
        print(f"LAMMPS input file '{filename}' has been created successfully!")
    return params


def create_lammps_data_pkl(element, dump_freq, run_step, dump_dir='./data/dumps', pd_data_dir='argon_data.pckl.gzip', verbose=False):
    """
    Note: Sample script for converting Lammps dump sequence data (dump.lj.*) to the .pkl format readable by ACE or PACE
    """
    from ase import Atoms
    from ase.io import read, write
    from ase.db import connect

    # All energies reference
    all_energies_ref = np.genfromtxt(os.path.join(dump_dir, 'etotal.txt'), skip_header=1)[:,1]

    # All dumps
    all_dumps = [os.path.join(dump_dir, f'dump.lj.{dump_freq * i}') for i in range(run_step // dump_freq + 1)]
    all_atoms, all_forces, all_energies = [], [], []

    # Read dumps and collect data
    for i, dump_file in enumerate(tqdm.tqdm(all_dumps)):
        atoms = read(dump_file, 0, format='lammps-dump-text')
        symbls = [element for i in range(len(atoms))]
        forces = atoms.get_forces()
        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell())
        energy = np.sum(atoms.arrays['c_pe_atom'])

        all_atoms.append(Atoms(symbols = symbls, positions = pos, cell = cell, pbc = True))
        all_forces.append(forces)
        all_energies.append(energy)
    
    # Test reference energies
    if verbose:
        all_energies_mae = np.abs(np.array(all_energies) - all_energies_ref)
        all_energies_rmse = np.sqrt(np.mean(all_energies_mae**2))
        print(f'Error in mean absolute energy: {all_energies_mae.mean():.2e}')
        print(f'Error in root mean square energy: {all_energies_rmse:.2e}')
        if all_energies_mae.mean() < 1e-6: print(f'\033[92mPASS\033[0m: Energy values are correct.')
        else: print(f'\033[91mFAIL\033[0m: Energy values are NOT correct.')

    # Save to pandas DataFrame
    data = {'energy': all_energies, 'forces': all_forces, 'ase_atoms': all_atoms, 'energy_corrected': all_energies}
    df = pd.DataFrame(data)
    df.to_pickle(pd_data_dir, compression='gzip', protocol=4)


def create_lammps_data_raw(element, dump_freq, run_step, dump_dir='./data/dumps', pd_data_dir='argon_data.pckl.gzip', verbose=False):
    """
    Note: Sample script for converting Lammps dump sequence data (dump.lj.*) to the .pkl format readable by ACE or PACE
    """
    from ase import Atoms
    from ase.io import read, write
    from ase.db import connect

    # All energies reference
    all_energies_ref = np.genfromtxt(os.path.join(dump_dir, 'etotal.txt'), skip_header=1)[:,1]

    # All dumps
    all_dumps = [os.path.join(dump_dir, f'dump.lj.{dump_freq * i}') for i in range(run_step // dump_freq + 1)]
    all_atoms, all_forces, all_energies = [], [], []
    file_nos = np.linspace(0, 2000000, 401, dtype=np.int32)

    # Read dumps and collect data
    for i, dump_file in enumerate(tqdm.tqdm(all_dumps)):
        atoms = read(dump_file, 0, format='lammps-dump-text')

        if i == 0:
            N_at = len(atoms)
            all_forces = np.zeros((len(file_nos),3*N_at))
            all_pos = np.zeros((len(file_nos),3*N_at))
            all_boxs = np.zeros((len(file_nos),9))
            at_type = np.zeros(N_at)

        forces = atoms.get_forces()
        pos = atoms.get_positions()
        cell = np.array(atoms.get_cell())
        all_forces[i,:] = forces.reshape(-1,)
        all_pos[i,:] = pos.reshape(-1,)
        all_boxs[i,:] = cell.reshape(-1,)

        energy = np.sum(atoms.arrays['c_pe_atom'])
        all_energies.append(energy)
    
    # Test reference energies
    if verbose:
        all_energies_mae = np.abs(np.array(all_energies) - all_energies_ref)
        all_energies_rmse = np.sqrt(np.mean(all_energies_mae**2))
        print(f'Error in mean absolute energy: {all_energies_mae.mean():.2e}')
        print(f'Error in root mean square energy: {all_energies_rmse:.2e}')
        if all_energies_mae.mean() < 1e-6: print(f'\033[92mPASS\033[0m: Energy values are correct.')
        else: print(f'\033[91mFAIL\033[0m: Energy values are NOT correct.')

    # Save to raw DataFrame
    np.savetxt('type.raw',at_type,fmt='%d')
    np.savetxt('force.raw',all_forces,fmt='%.8f')
    np.savetxt('box.raw',all_boxs,fmt='%.8f')
    np.savetxt('coord.raw',all_pos,fmt='%.8f')
    np.savetxt('energy.raw',energy,fmt='%.8f')
    fid = open('type_map.raw','w')
    fid.write(element)
    fid.close()



""" Visualization Utility Functions, Saúl Eduardo Pérez Herrera, July 2025 """
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


def visualize_lammps_data():
    import os
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import hist
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import norm

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
    plt.savefig('./data/figs/energy_distribution.png')
    plt.close()

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

    plt.savefig('./data/figs/velocities_over_time.png')
    plt.close()



