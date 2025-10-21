import numpy as np
import mod_qcontrol as qc
import jax
import jax.numpy as jnp
import argparse
from functools import partial
from mpi4py import MPI # type: ignore
import os

comm = MPI.COMM_WORLD
size = comm.Get_size() #number of processes
rank = comm.Get_rank() # ranks each process



parser = argparse.ArgumentParser()
parser.add_argument('number', type=int, help='number of spins')
parser.add_argument('spin_import', type=float, help='mean spin matters')
args = parser.parse_args()

n=args.number #system size
para_per_gate = 3 #number of parameters per control gate
div = 3 #number of groups in the initial conditions
c = args.spin_import # 0 if you do not care about the spin lenght. Otherwise, 1 is a good value.
a_max = 4.0 # the max value of the X and Z pulses
one=1e-1 #the min value of the parameter for H0
constrain_lower=jnp.array([-a_max, 0.0 , 0.0])
constrain_upper=jnp.array([a_max, jnp.sqrt(a_max), 1.0])
ini_params= jax.random.uniform(jax.random.PRNGKey(rank*n*89), minval=constrain_lower[:, None], maxval=constrain_upper[:, None], shape=(3, div))
ini_params = jnp.repeat(ini_params,int(para_per_gate/div), axis=-1)


total_t = 8. #total duration of the protocol when Z,X and H0 parameters are optimized 

learn_params = {
    'learning_steps': 400000, #max number of steps
    'ini_learning_rate' : 0.04, # initial learning rate
    'patience': 400, # steps the algorithm will wait to reduce the learning rate if the cost function do not decrease
    'spin_lenght': c, #desired spin length
    'spin_focus': 0., #how much do you care about the spin lenght, 0=>Not at all
    'first_smooth': 0.,#1.8/para_per_gate, # First derivative hyperparameter
    'second_smooth': 0., #0.21/para_per_gate, # Second derivative hyperparameter
    'duration': 0.005, # duration hyperparameter
    'limit_amplitude':2.*a_max/one  # limit of the amplitude when X and Z paramters are transformed
}
#ini_params = jax.random.truncated_normal(jax.random.PRNGKey(rank*para_per_gate), lower=constrain_lower[:, None], upper=constrain_upper[:, None], shape=(3,para_per_gate))
print('loading...')
#here we load the projected operators
h0 = np.load("/home/ipht/ecarrera/qcontrol/codes/operators/h0_"+str(n)+".npy") #Hamiltonian
spins = np.load("/home/ipht/ecarrera/qcontrol/codes/operators/spin_ops_"+str(n)+".npy") #Control operators: collective Z and X
ini = np.load('/home/ipht/ecarrera/qcontrol/codes/operators/ini_state_'+str(n)+'.npy') #the initial state
sq_d = jnp.array([0.,1.,0.]) # Squeezing direction
mean_d = jnp.array([0.,0.,1.]) # Mean spin direction

##
print('Ready to optimize!')
mem,ps,sss,idsa, leng = qc.jax_control_squeezing_mod(h0, spins, sq_d,mean_d, ini, total_t, n, ini_params, learn_params)


all_mem = comm.gather(mem, root=0)
all_ps = comm.gather(ps, root=0)
all_sss = comm.gather(sss, root=0)
all_idsa = comm.gather(idsa, root=0)
all_leng = comm.gather(leng, root=0)

if rank==0:
    print(all_mem)
    filepath2 = '/home/ipht/ecarrera/qcontrol/data/itneverends/'+str(n)+'spins/final_parameters_sqy_mz_N'+str(n)+'_ising_T'+str(total_t)+'ysq_params'+str(para_per_gate)+'div'+str(div)+'c'+str(c)+'_end00029.txt'
    with open(filepath2, 'w') as f2:
        f2.write('Optimization Parameters:\n')
        for key, value in learn_params.items():
            f2.write(f'{key}: {value}\n')
        f2.write('\n')
        f2.write('Final results\n')
        for i in range(size):
            f2.write('Best squeezing, steps, dif, Spin length\n')
            f2.write(f'{all_mem[i]} {all_sss[i]} {all_idsa[i]} {all_leng[i]}\n')
            storing_final_params2 = np.column_stack((np.linspace(0, total_t, para_per_gate), all_ps[i].T))
            np.savetxt(f2, storing_final_params2, fmt='%.5f', header="time, Z-control, X-control, H0")