import numpy as np, os, sys, argparse
from python.utils import write_lammps_input_parameterized, create_lammps_data_pkl



parser = argparse.ArgumentParser(description="Generate LAMMPS Dataset")
parser.add_argument("--filename", type=str, default="create_dataset.in", help="Output filename")
parser.add_argument("--element", type=str, default='Ar', help="Element type for LAMMPS")
parser.add_argument("--pair_style", type=str, default='lj/cut', help="Pair style for LAMMPS")
parser.add_argument("--epsilon", type=float, default=0.01032, help="Epsilon value for LJ potential (eV)")
parser.add_argument("--sigma", type=float, default=3.4, help="Sigma value for LJ potential (Angstrom)")
parser.add_argument("--cutoff", type=float, default=8.5, help="Cutoff distance for LJ potential (Angstrom)")
parser.add_argument("--skin", type=float, default=2.0, help="Skin distance for neighbor list (Angstrom)")
parser.add_argument("--num_replicate", type=int, default=2, help="Number of replicates in the simulation box")
parser.add_argument("--cell_param", type=float, default=5.00, help="Cell parameter for the simulation box (Angstrom)")
parser.add_argument("--temp_obj", type=float, default=10, help="Target temperature for the simulation (Kelvin)")
parser.add_argument("--temp_init", type=float, default=10, help="Initial temperature for the simulation (Kelvin)")
parser.add_argument("--temp_final", type=float, default=10, help="Final temperature for the simulation (Kelvin)")
parser.add_argument("--press_obj", type=float, default=137.3, help="Target pressure for the simulation (bar)")
parser.add_argument("--time_step", type=float, default=0.01078, help="Time step for the simulation (picoseconds)")
parser.add_argument("--num_step", type=int, default=200000, help="Total number of steps in the simulation")
parser.add_argument("--run_step", type=int, default=200000, help="Number of steps to run in the simulation")
parser.add_argument("--dump_freq", type=int, default=100, help="Frequency of dump output (steps)")
parser.add_argument("--log_freq", type=int, default=200, help="Frequency of log output (steps)")
parser.add_argument("--npt_thermo_freq", type=float, default=0.02, help="Thermodynamic output frequency for NPT ensemble (seconds)")
parser.add_argument("--nvt_thermo_freq", type=float, default=0.02, help="Thermodynamic output frequency for NVT ensemble (seconds)")
parser.add_argument("--npt_baro_freq", type=float, default=0.2, help="Barostat output frequency for NPT ensemble (seconds)")
parser.add_argument("--num_chains", type=int, default=5, help="Number of chains in the simulation")
parser.add_argument("--num_mtk", type=int, default=5, help="Number of MTK (Molecular Thermodynamic Kinetics) steps")
parser.add_argument("--random_seed", type=int, default=2025, help="Random seed for reproducibility")
parser.add_argument("--mode", type=str, default='create', help="Mode for the simulation (write or run or create)")
parser.add_argument("--verbose", action='store_true', help="Enable verbose output")

parser.add_argument("--dump_dir", type=str, default="./data/dumps", help="Output directory for dump files")
parser.add_argument("--pd_data_dir", type=str, default='argon_data.pckl.gzip', help="Output directory for pandas data files")
args = parser.parse_args()



if args.mode == 'write':
    write_lammps_input_parameterized(
        filename=args.filename,
        pair_style=args.pair_style,
        epsilon=args.epsilon,
        sigma=args.sigma,
        cutoff=args.cutoff,
        skin=args.skin,
        num_replicate=args.num_replicate,
        cell_param=args.cell_param,
        temp_obj=args.temp_obj,
        temp_init=args.temp_init,
        temp_final=args.temp_final,
        press_obj=args.press_obj,
        time_step=args.time_step,
        num_step=args.num_step,
        run_step=args.run_step,
        dump_freq=args.dump_freq,
        log_freq=args.log_freq,
        npt_thermo_freq=args.npt_thermo_freq,
        nvt_thermo_freq=args.nvt_thermo_freq,
        npt_baro_freq=args.npt_baro_freq,
        num_chains=args.num_chains,
        num_mtk=args.num_mtk,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )




elif args.mode == 'run':
    # TODO: Use subprocess to run the LAMMPS simulation on a GPU node
    pass



elif args.mode == 'create':
    create_lammps_data_pkl(
        element=args.element,
        dump_freq=args.dump_freq,
        run_step=args.run_step,
        dump_dir=args.dump_dir,
        pd_data_dir=args.pd_data_dir,
        verbose=args.verbose,
    )
