from ase import Atoms
from ase.io import read, write
from ase.db import connect
import numpy as np, tqdm, pandas as pd



element = 'Ar'



all_energies_ref = np.genfromtxt('../lammps/etotal.txt',skip_header=1)[:,1]



all_dumps = [f'../lammps/dump.lj.{100 * i}' for i in range(2001)]
all_atoms = []
all_forces = []
all_energies = []


for i,dump_file in enumerate(tqdm.tqdm(all_dumps)):
    atoms = read(dump_file, 0, format='lammps-dump-text')
    symbls = [element for i in range(len(atoms))]
    forces = atoms.get_forces()
    pos = atoms.get_positions()
    cell = np.array(atoms.get_cell())
    energy = np.sum(atoms.arrays['c_pe_atom'])

    all_atoms.append(Atoms(symbols = symbls, positions = pos, cell = cell, pbc = True))
    all_forces.append(forces)
    all_energies.append(energy)



data = {'energy': all_energies, 'forces': all_forces, 'ase_atoms': all_atoms, 'energy_corrected': all_energies}
df = pd.DataFrame(data)
df.to_pickle('my_data..pckl.gzip', compression='gzip', protocol=4)