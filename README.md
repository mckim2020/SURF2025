# SURF2025
2025 Summer Undergraduate Research Fellowship (SURF)



## Workflow
### Write LAMMPS Input File
```
python create_dataset.py --mode write
```

LAMMPS input script ```create_dataset.in``` will be created in the current directory.

### Run LAMMPS
```
sbatch run_gpu.sbatch
```

After running lammps, ```dump.lj.*``` format files will be saved into ```data/dumps``` directory.

### Process LAMMPS Output File
```
python create_dataset.py --mode create
```

This will create training files such as ```argon_data.pckl.gzip``` in the current directory.



## Developers
PI: Prof. Wei Cai  
Mentor: Myung Chul Kim  
Mentee: Saúl Eduardo Pérez Herrera