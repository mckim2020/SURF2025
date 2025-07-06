# SURF2025
2025 Summer Undergraduate Research Fellowship (SURF)



## Workflow
### Write LAMMPS Input File
```
python create_dataset.py --mode write
```

### Run LAMMPS
```
sbatch run_gpu.sbatch
```

### Process LAMMPS Output File
```
python create_dataset.py --mode create
```



## Developers
PI: Prof. Wei Cai  
Mentor: Myung Chul Kim  
Mentee: Saúl Eduardo Pérez Herrera