import dynamiqs as dq
import jax.numpy as jnp
import jax
import optax
from jax.scipy.linalg import expm
from functools import partial
from itertools import product
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
cmap = plt.cm.Spectral #colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import numpy as np
#CONSTRAINS FOR THE OPTIMIZATION
one=1e-1

### For collective operators
def pauli_matrix(a):
    #pauli_dict = {'X': 0,'Y': 1,'Z': 2, 'I': 3} The meaning
    matrices = jnp.array([ [[0, 1.], [1., 0]],  # X
            [[0, -1j], [1j, 0]],  # Y
            [[1., 0], [0, -1.]],   # Z
            [[1., 0], [0, 1.]]], dtype=jnp.complex64)   # I
            
    return matrices[a]

@jax.jit
def kron_all(ops):
    a=ops[0]
    for i in range(1,ops.shape[0]):
        a=jnp.kron(a,ops[i])
    return a

@partial(jax.jit, static_argnames=['n'])
def col_sigmas(n, pauli):
    def body(i,p0):
        return p0+sigma_j(i,pauli,n)
    return jax.lax.fori_loop(1,n, body, sigma_j(0,pauli,n))

@partial(jax.jit, static_argnames=['n'])
def sigma_j(j,a,n):
    ops = jnp.repeat(jnp.eye(2, dtype=jnp.complex64).reshape(1,2,2), n, axis=0)
    ops = ops.at[j].set(pauli_matrix(a))
    return kron_all(ops)

@partial(jax.jit, static_argnames=['n'])
def sigmaA_jxsigmaB_k(j,k,A,B,n): #A=!B
    ops = jnp.repeat(jnp.eye(2, dtype=jnp.complex64).reshape(1,2,2), n, axis=0)
    ops = ops.at[j].set(pauli_matrix(A))
    ops = ops.at[k].set(pauli_matrix(B))
    return kron_all(ops)


def generate_basis(N): # 0 is up, 1 is down
    """Generate computational basis states for an N-spin system."""
    return jnp.array([list(state) for state in product([0, 1], repeat=N)])

def translation_operator(N): #shifts to the left
    """Construct the translation operator for an N-spin periodic chain."""
    basis = generate_basis(N)
    return  jnp.where(jax.vmap(lambda x: jnp.all(basis==jnp.roll(x, shift=-1), axis=1))(basis), 1, 0).T


def get_momentum_eigenstates(N,k_index): # ujik@real_vander_ising_H(3,ring(3))@ujik.T
    eigvals, eigvecs = jnp.linalg.eig(translation_operator(N))
    k_values = jnp.exp(2j * jnp.pi * jnp.arange(N) / N)
    target_eigval = k_values[k_index]
    matching_indices = jnp.where(jnp.isclose(eigvals, target_eigval, atol=1e-6))[0]
    return (eigvecs[:, matching_indices].T).astype(jnp.complex64)

def sym_sigma_j(j,a,n, temp):
    
    return temp@sigma_j(j,a,n)@temp.T

def sym_col_sigmas(n, pauli, temp):
    result = sym_sigma_j(0, pauli, n, temp)  # Initialize with first term
    for i in range(1, n):
        result += sym_sigma_j(i, pauli, n, temp)
    return result


@partial(jax.jit, static_argnames=['N'])
def real_vander_ising_H(N, vec_pos,C=1.,b=1.):
    
    return sum([(C/b**6)*((1/jnp.linalg.norm(vec_pos[i]-vec_pos[j]))**6.0)*(sigmaA_jxsigmaB_k(i,j,2,2,N)+sigma_j(i,2,N)+sigma_j(j,2,N)) for i in range(N-1) for j in range(i+1,N)])     

@partial(jax.jit, static_argnames=['N'])
def ring(N):
    def vec_pos(m,N):
        r=0.5/jnp.sin(jnp.pi/N)
        return r*jnp.array([jnp.sin(m*2*jnp.pi/N), jnp.cos(m*2*jnp.pi/N)])
    return vec_pos(jnp.arange(N),N).T

@partial(jax.jit, static_argnames=['N'])
def square(N,b=1.):
    L = int(jnp.sqrt(N))
    if L * L != N:
        raise ValueError("N must be a perfect square for a square lattice.")
    x_coords, y_coords = jnp.meshgrid(jnp.arange(L), jnp.arange(L), indexing='ij')
    coordinates = jnp.column_stack((x_coords.ravel(), y_coords.ravel()))
    return b*coordinates

@partial(jax.jit, static_argnames=['N'])
def vander_ising_H(N,r,C=1.,b=1.):
    
    return sum([(C/b**6)*((1/jnp.linalg.norm(r[i]-r[j]))**6.0)*sigmaA_jxsigmaB_k(i,j,2,2,N) for i in range(N-1) for j in range(i+1,N)])    
    
def square(l):
    n=l*l
    a = jnp.arange(n).reshape(l,l)
    def get_pos(x):
        return jnp.array(jnp.where(a==x)).reshape(-1)
    return get_pos

@partial(jax.jit, static_argnames=['n'])
def dipolar_xy(n,r,C=1.): #n=l*l
    b=jnp.linalg.norm(r[0]-r[1])
    return sum([C*((b/jnp.linalg.norm(r[i]-r[j]) )**3.0)*(sigmaA_jxsigmaB_k(i,j,0,0,n) + sigmaA_jxsigmaB_k(i,j,1,1,n)) for i in range(n-1) for j in range(i+1,n)])


@jax.jit
def vOv(op_m, states_t):
    return jnp.real(jnp.einsum('bij,bij->b', jnp.conjugate(states_t), op_m@states_t))

@jax.jit
def varO(op, states_t):
    oket = op@states_t
    return jnp.real(jnp.einsum('bij,bij->b', jnp.conjugate(oket), oket) - jnp.einsum('bij,bij->b', jnp.conjugate(states_t), oket)**2)

@jax.jit
def wineland(states_t, On, On_orth, n):
    sn = vOv(On, states_t)
    var = vOv(On_orth@On_orth, states_t) - vOv(On_orth, states_t)**2
    return jnp.real(n*var /(sn**2)) 



@jax.jit #time evolution
def time_evo(b, T, H_int,control, vec):
    dt = T/b.shape[-1]
    def loop_body(i, new_vec):
        ups = expm(-1.j * dt * ( (b[2][i]**2+one)*H_int + b[0][i]*control[0] + b[1][i]**2 *control[1]))
        return ups@new_vec

    fin = jax.lax.fori_loop(0, b.shape[-1], loop_body, vec)
    fin = jnp.expand_dims(fin/jnp.linalg.norm(fin),axis=0) 
    return fin

def css(theta, phi,n): #coherent states
    state_list = [jnp.cos(theta/2)*dq.basis(2, 0)+jnp.sin(theta/2)*jnp.exp(1.j*phi)*dq.basis(2, 1)]*n
    return dq.tensor(*state_list)

@jax.jit
def sn(vec,spin_ops):
    return vec[0]*spin_ops[0] + vec[1]*spin_ops[1] + vec[2]*spin_ops[2]

@partial(jax.jit, static_argnames=['n'])
def wineland_cost(b, T,H0,spin_ops, sq_d, mean_d,psi0, n, c, a=1., g=2.):
    '''
    b: Parameters
    T: final time
    spin_ops: array of the spin operators: x , y and z
    mean_d: angles, theta and phi indicating the mean spin direction
    sq_d: angles indicating the squeezing direction
    psi_0: initial state, n: system size, c: spin length imporantance
    '''
    fin = time_evo(b, T, H0,[spin_ops[2],spin_ops[0]], psi0) #control: Z and X
    var_sq = varO(sn(sq_d,spin_ops), fin)[0]
    jmean = vOv(sn(mean_d,spin_ops), fin)[0]
    j1=vOv(sn(sq_d,spin_ops), fin)[0]
    j2=vOv(sn(jnp.linalg.cross(sq_d,mean_d),spin_ops), fin)[0]
    return n*var_sq/((jmean)**2) + a*((j1/n)**2+(j2/n)**2) + g*jnp.abs(c-(jmean/n)**2)

@partial(jax.jit, static_argnames=['n'])
def wineland_and_length(b, T,H0,spin_ops, sq_d, mean_d,psi0, n):
    '''
    b: Parameters
    T: final time
    spin_ops: array of the spin operators: x , y and z
    mean_d: angles, theta and phi indicating the mean spin direction
    sq_d: angles indicating the squeezing direction
    psi_0: initial state, n: system size, c: spin length imporantance
    '''
    fin = time_evo(b, T, H0,[spin_ops[2],spin_ops[0]], psi0) #control: Z and X
    var_sq = varO(sn(sq_d,spin_ops), fin)[0]
    jmean = vOv(sn(mean_d,spin_ops), fin)[0]
    
    return n*var_sq/((jmean)**2), jmean/n


@jax.jit
def penalties(params, a, b,c, lim): # 2.e-3/params.shape[-1], 6.e-2/params.shape[-1]
    def first_derivative(a, time):
        return jnp.sum(((a[1:] - a[:-1])/(time[:-1]))**2)  #jnp.sum((a[2:]-2*a[1:-1]+a[:-2])**2)
    def second_derivative(a,time):
        return jnp.sum(((a[2:]-a[1:-1])/(time[2:])*(time[:-2])+(a[:-2]-a[1:-1])/(time[:-2])**2)**2)

    xpa = params[1]**2/(params[2]**2 +one)
    zpa = params[0]/(params[2]**2 +one)
    smoothness1 = first_derivative(xpa/lim,params[2]**2 +one) + first_derivative(zpa/lim,params[2]**2 +one)
    smoothness2 = second_derivative(xpa/lim,params[2]**2 +one) + second_derivative(zpa/lim,params[2]**2 +one) 
    #fluctuations = jnp.std(xpa/xpa_norm) + jnp.std(zpa/zpa_norm)
    amplitud_x = jnp.sum(jnp.maximum(0.0,  xpa -lim)**2)
    amplitud_z = jnp.sum(jnp.maximum(0.0,  zpa -lim)**2) + jnp.sum(jnp.maximum(0.0,  -lim - zpa )**2)
    duration = jnp.sum(params[2]**2)
    return a*smoothness1 + b*smoothness2 + c*duration + 0.1*(amplitud_x + amplitud_z) 




def funs_with_penalty(params, duration, h0,spins, sq_d, mean_d, ini_state, npart,imp,focus,a,b,c,lim):
    return wineland_cost(params, duration, h0,spins, sq_d, mean_d, ini_state, npart,imp,g=focus) + penalties(params, a,b,c,lim) 



def get_new_time_and_params(params, total_t): #transforms the parameters
    para_per_gate = params.shape[-1]
    dt = total_t/para_per_gate
    new_dt = dt*(params[2]**2 +one)
    new_time = jnp.concatenate([jnp.array([0]),  jnp.cumsum(new_dt)])
    #new_time = new_time[:-1]
    return new_time, params[0]/(params[2]**2 +one), params[1]**2/(params[2]**2 +one) #time, Zpulses, Xpulses


def find_angle(liststa,n, op_main, op2, op3, time):
    points = jnp.append(jnp.linspace(0., jnp.pi/2, 150), jnp.linspace(jnp.pi/2, jnp.pi, 150, endpoint=True))
    def one_point(theta):
        u = n*varO(jnp.cos(theta)*op2 + jnp.sin(theta)*op3,liststa)/vOv(op_main,liststa)**2
        arg_km = jnp.argmin(u)
        return arg_km, u[arg_km] #for a fixed angle we get at what time we get the min wineland
    args, squ = jax.vmap(one_point)(points) # vectorized to many angles
    o=jnp.argmin(squ) #which angle gives the lowest wineland
    angle_opti = points[o]
    time_where = time[args[o]] 
    return squ[o], angle_opti, time_where 




def jax_control_squeezing_mod(h0,spins, sq_dir, mean_spin, ini, duration, npart, params0, lp):
    
    def cost(ps):
        return funs_with_penalty(ps, duration, h0,spins, sq_dir, mean_spin, ini, npart,lp['spin_lenght'],lp['spin_focus'],lp['first_smooth'],lp['second_smooth'], lp['duration'], lp['limit_amplitude'])


    @jax.jit
    def train_step(params, opt_state):
        value, grad = jax.value_and_grad(cost)(params)
        updates, opt_state = opt.update(grad, opt_state, params, value=value)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, value
    
    opt = optax.chain(optax.adam(lp['ini_learning_rate']), optax.contrib.reduce_on_plateau(patience=lp['patience'],factor=0.5, rtol=1e-3, cooldown=10, min_scale=1e-2))
    opt_state = opt.init(params0)


    ss = 0
    counter=1
    dif=1.
    
    best_loss = jnp.inf
    best_loss_prev = jnp.inf
    best_params = params0
    wait = 10000 # if no improvement then stop
    
    def body(state):
        
        params0, opt_state, ss, best_loss,best_loss_prev, best_params, counter, dif = state
        params0, opt_state, sq = train_step(params0, opt_state)

        is_better = sq < best_loss
        
        best_loss_prev = jnp.where(is_better, best_loss, best_loss_prev)
        best_loss = jnp.where(is_better, sq, best_loss)
        dif = best_loss_prev - best_loss
        best_params = jnp.where(is_better, params0, best_params)
        counter = jnp.where(is_better, 0, counter+1)
        
        return params0, opt_state, ss + 1, best_loss,best_loss_prev, best_params, counter, dif

    initial_state = (params0, opt_state, ss, best_loss, best_loss_prev, best_params, counter, dif)
   
    def cond(state):
        _, _, ss, _, _, _,counter, dif = state
        return jnp.logical_and(ss < lp['learning_steps'], jnp.logical_and(dif>1e-8, counter<wait))

    final_state = jax.lax.while_loop(cond, body, initial_state)
    params0, opt_state, ss, best_loss, best_loss_prev, best_params, counter, dif = final_state
    squeezing =wineland_cost(best_params, duration, h0,spins, sq_dir, mean_spin, ini, npart, 0., 0., 0.)
    spin_lenght = wineland_cost(best_params, duration, h0,spins, sq_dir, mean_spin, ini, npart, 0., 0.,1.) - squeezing
    return squeezing, best_params ,ss, dif, jnp.sqrt(spin_lenght)  #final squeezing, best parameters, number of steps used, difference between the last two best costs


        

def entropy_half(psi,a,b): #shape m,2**n
    #n = int(jnp.log2(psi.shape[-1]))
    #a = 2**(n - int(n/2))
    ma = psi.reshape((psi.shape[0], a, b))
    u,s,vd=jnp.linalg.svd(ma, full_matrices=False)
    s2 = s*s
    return -jnp.sum(s2 *jnp.where(s2 != 0,jnp.log(s2),0.) , axis=1)          



def control_tevol(params, time_steps, control, h0,ini_state):
    simu_steps = 1000
    simu_time = jnp.linspace(0., time_steps[-1], simu_steps)
    #time_steps = jnp.linspace(0., duration, len(params[0])+1)
    hcz = dq.pwc(time_steps, params[0], control[0]) #Z
    hcx = dq.pwc(time_steps, params[1], control[1]) #X
    #h0c = dq.pwc(time_steps, (1+params[2]**2), h0)
    going = dq.sesolve(h0 + hcz + hcx, ini_state, simu_time, options=dq.Options(progress_meter=None, save_states=True)) #solver=dq.solver.Kvaerno3(rtol=1e-6, atol=1e-6, max_steps=int(1e8))
    return going

@jax.jit
def dissipation_test(params, time_steps, control, jump_ops, h0, rho0):
    simu_steps = 2
    simu_time = jnp.linspace(0., time_steps[-1], simu_steps)
    hcz = dq.pwc(time_steps, params[0], control[0])
    hcx = dq.pwc(time_steps, params[1], control[1])
    
    going = dq.mesolve(h0 + hcz + hcx, jump_ops, rho0, simu_time ,options=dq.Options(progress_meter=None, save_states=True), solver=dq.solver.Dopri8()) #safety_factor=0.8, min_factor=0.01, max_factor=2.0, atol=1e-7
    return going

@jax.jit
def dissipation_test_nocontrol(final_time, jump_ops, h0, rho0):
    simu_steps = 2
    simu_time = jnp.linspace(0., final_time, simu_steps)
    going = dq.mesolve(h0, jump_ops, rho0, simu_time ,options=dq.Options(progress_meter=None, save_states=True), solver=dq.solver.Dopri8())
    return going

@jax.jit
def vOv_rho(op, rh0_states_t):
    return jnp.real(jnp.trace(jnp.matmul(rh0_states_t, op), axis1=1, axis2=2))

@jax.jit
def find_angle_rh0(liststa, n, op_main, op2, op3, time):
    points = jnp.linspace(0., jnp.pi, 80)
    
    def one_point(theta):
        u = n*(vOv_rho(liststa, (jnp.cos(theta)*op2 + jnp.sin(theta)*op3)@(jnp.cos(theta)*op2 + jnp.sin(theta)*op3)) - vOv_rho(liststa, jnp.cos(theta)*op2 + jnp.sin(theta)*op3)**2)/vOv_rho(liststa, op_main)**2
        arg_km = jnp.argmin(u)
        return arg_km, u[arg_km] 
    args, squ = jax.vmap(one_point)(points)
    o=jnp.argmin(squ)
    angle_opti = points[o]
    time_where = time[args[o]]
        
    return squ[o], angle_opti, time_where


@jax.jit
def find_var(liststa,vec2, vec3, spin_list):
    points = jnp.append(jnp.linspace(0., jnp.pi/2, 100), jnp.linspace(jnp.pi/2, jnp.pi, 100, endpoint=True))
    def one_point(theta):
        u = varO(sn(jnp.cos(theta)*vec2 + jnp.sin(theta)*vec3, spin_list),liststa)
        return u[0]
    squ = jax.vmap(one_point)(points)
    o=jnp.nanargmin(squ)
    angle_opti = points[o]
    #time_where = time[args[o]] 
    return squ[o], angle_opti#, time_where 

@jax.jit
def find_sq(res_low, spin_ops, num_theta=200, num_phi=200):
    # 1. build 1D grids of angles
    u = jnp.linspace(0.0, jnp.pi, num_theta)
    v = jnp.linspace(0.0, 2*jnp.pi, num_phi, endpoint=False)

    # 2. mesh and flatten into pairs (θ,φ)
    ug, vg = jnp.meshgrid(u, v, indexing='ij')        # shape (num_theta, num_phi)
    pairs   = jnp.stack([ug.ravel(), vg.ravel()], -1)  # shape (num_theta*num_phi, 2)

    # 3. define overlap function for a single (θ,φ)
    @jax.jit
    def overlap_fn(tp):
        t, p = tp
        v = jnp.array([jnp.sin(t)*jnp.cos(p),
                       jnp.sin(t)*jnp.sin(p),
                       jnp.cos(t)])  # unit vector
        return jnp.abs(vOv(sn(v, spin_ops), res_low.states.to_jax()))  # shape (n_states,)

    # 4. vmap over all grid points → (num_theta*num_phi, n_states)
    all_ov = jax.vmap(overlap_fn)(pairs)

    # 5. per‐state best index & its value
    argmax_flat = jnp.argmax(all_ov, axis=0)  # (n_states,)
    best_val    = jnp.max(all_ov,   axis=0)   # (n_states,)

    # 6. unravel flat index → (iθ, iφ) for each state
    iu, iv    = jnp.unravel_index(argmax_flat, (num_theta, num_phi))
    t_opt     = u[iu]  # (n_states,)
    p_opt     = v[iv]  # (n_states,)

    # 7. build orthonormal frame in batch
    su, cu = jnp.sin(t_opt), jnp.cos(t_opt)
    sv, cv = jnp.sin(p_opt), jnp.cos(p_opt)

    x_p = jnp.stack([su * cv, su * sv, cu], axis=-1)       # “mean” direction
    e1  = jnp.stack([cu * cv, cu * sv, -su], axis=-1)
    e2  = jnp.cross(x_p, e1)

    # 8. compute variance & squeezing angle in batch
    @jax.jit
    def var_and_angle(ep1, ep2, state):
        v, ang = find_var(jnp.expand_dims(state,0), ep1, ep2, spin_ops)
        return v, ang

    vars, sqang = jax.vmap(var_and_angle)(e1, e2, res_low.states.to_jax())
    sqang = sqang.reshape(-1,1)

    # 9. final squeezed‐direction
    sq_dir = jnp.cos(sqang)*e1 + jnp.sin(sqang)*e2

    return vars, x_p, sq_dir, best_val

def sample_from_distribution(probs, key, num_samples): #direct sampling
    cdf = jnp.cumsum(probs)
    random_values = jax.random.uniform(key, shape=(num_samples,))
    sampled_indices = jnp.searchsorted(cdf, random_values, side='right')
    return sampled_indices

def jacky(data_n, data_or,n): #jackknife estimation of the Wineland parameter
    M=data_n.size
    ys = (jnp.sum(data_n/(M-1)) - data_n/(M-1))
    var_i_j = ((jnp.sum((data_or**2)/(M-1)) -(data_or**2)/(M-1)) -((jnp.sum(data_or/(M-1))- data_or/(M-1)))**2)*((M-1)/(M-2))
    xi_j = n*var_i_j/(ys*ys)
    mean_jacky = jnp.mean(xi_j)
    std_jacky=jnp.std(xi_j)*jnp.sqrt(data_n.size -1)
    return mean_jacky, std_jacky #mean value and standard deviation

@jax.jit
def g_obs(a,n,data_orthdir, data_ndir):
    return n*jnp.mean(data_orthdir**2) + n*a[0]**2 -2*a[0]*n*jnp.mean(data_orthdir) + a[1]**2 - 2*a[1]*jnp.abs(jnp.mean(data_ndir))



@jax.jit
def p_value_upper(a,n,data_orthdir, data_ndir):
    M=data_ndir.size
    gc1 = n*(1+jnp.abs(a[0]))**2 + (1+jnp.abs(a[1]))**2 - 1
    gc0 = (1-jnp.abs(a[1]))**2 -1
    g  = g_obs(a,n,data_orthdir, data_ndir)
    return jnp.exp((M*g**2)/(2*gc1*gc0 + 2*(gc1-gc0)*g/3))




def p_value_upper_optimizer(data_orthdir, data_ndir, n, inia):
    constrain_lower = jnp.array([jnp.mean(data_orthdir/n),0.5])
    constrain_upper = jnp.array([0.1,1.])
    @jax.jit
    def cost(a):
        return p_value_upper(a,n,data_orthdir, data_ndir) + 10000*jnp.maximum(0.,g_obs(a,n,data_orthdir, data_ndir))**2 
    
    @jax.jit
    def train_step(params, opt_state):
        value, grad = jax.value_and_grad(cost)(params)
        updates, opt_state = opt.update(grad, opt_state, params, value=value)
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(params, lower=constrain_lower, upper=constrain_upper)
        return params, opt_state

    opt = optax.adam(0.04)
    opt_state = opt.init(inia)

    for _ in range(8000):
        inia, opt_state= train_step(inia, opt_state)
        
    return inia, p_value_upper(inia,n,data_orthdir, data_ndir)




def q_husumi(psi, n, thetas, phis):
    
    def one_direction(theta, phi):
        return (jnp.abs(jnp.vdot(css(theta,phi,n).to_jax(), psi))**2)/jnp.pi
    
    return jax.vmap(jax.vmap(one_direction, in_axes=(0, None)), in_axes=(None, 0))(thetas, phis)

def plot_q_husumi(psi,n,axis1, axis2, limtheta=[0.,np.pi], limphi=[-np.pi, np.pi]):  
    theta_values = np.linspace(limtheta[0],limtheta[1],500)
    phi_values = np.linspace(limphi[0],limphi[1],500)


    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)
    x_grid = jnp.cos(phi_grid) * jnp.sin(theta_grid)
    y_grid = jnp.sin(phi_grid) * jnp.sin(theta_grid)
    z_grid = jnp.cos(theta_grid)
    poss = {'x':x_grid, 'y':y_grid, 'z':z_grid}
    
    # Plot the Q Husimi function in a circular plot
    plt.figure(figsize=(8,8))
    plt.contourf(poss[axis1], poss[axis2], q_husumi(psi,n,theta_values, phi_values), cmap='Spectral', levels=20)
    plt.colorbar(label='Q Husimi Function')
    #plt.title('Q Husimi Function for Spin State (Spherical Projection)')
    plt.xlabel(axis1, fontsize=18)
    plt.ylabel(axis2, fontsize=18)
    plt.ylim([-1,1])
    plt.xlim([-1,1])
    
    plt.gca().set_aspect('equal')  # Ensure the aspect ratio is equal for circular appearance 



def plot_q_husumi_3d(ax, psi, n, view_elev, view_azim):
    limtheta = [0., np.pi] 
    limphi = [-np.pi, np.pi]
    
    # Create the grid
    theta_values = np.linspace(limtheta[0], limtheta[1], 500)
    phi_values = np.linspace(limphi[0], limphi[1], 500)
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values)
    
    # Convert to Cartesian coordinates
    x_grid = jnp.cos(phi_grid) * jnp.sin(theta_grid)
    y_grid = jnp.sin(phi_grid) * jnp.sin(theta_grid)
    z_grid = jnp.cos(theta_grid)
    
    # Calculate the Husimi Q function
    husu = q_husumi(psi, n, theta_values, phi_values)
    husu_normalized = husu / jnp.max(husu)  # Normalize for color mapping
    #cmap = cm.viridis  # Choose a colormap
     # Define levels for the colorbar
    num_levels = 6
    levels = np.linspace(0, jnp.max(husu), num_levels)
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

    # Plot the 3D surface in the provided Axes object
    surface = ax.plot_surface(
        x_grid, y_grid, z_grid, 
        facecolors=cmap(husu_normalized), #norm(husu)
        rstride=5, cstride=5, alpha=0.9
    )
    
    # Colorbar setup with controlled levels
    mappable = plt.cm.ScalarMappable(cmap=cmap) #, norm=norm
    mappable.set_array(husu)
    #cbar = plt.colorbar(mappable, ax=ax, shrink=0.3, pad=0.05, orientation='vertical')
    
    #cbar.ax.tick_params(labelsize=5)  # Adjust font size of color bar ticks)
    #cbar.set_label("Husimi Q Function", rotation=90, labelpad=10)
    
    # View settings
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])



    
def extract_data(filepath1):
    sq1 = []
    ps1 = []
    dif = []
    ln=[]
    c=10000
    with open(filepath1, 'r') as file:
        lines = file.readlines()
        for num, line in enumerate(lines):
            if line == 'Best squeezing, steps, dif, Spin length\n':
                a = lines[num + 1].strip()
                sq1.append(float(a.split()[0]))
                ln.append(float(a.split()[-1]))
                dif.append(float(a.split()[2]))
            if line == '# time, Z-control, X-control, H0\n':
                j = 1 
                firs = []
            # Ensure you don't exceed the bounds of the lines list
                while (num + j  < len(lines) and lines[num + j ] != 'Best squeezing, steps, dif, Spin length\n' and len(firs) < c):
                    b = lines[num + j].strip()
                    b = b.split()
                    firs.append([float(b[1]), float(b[2]), float(b[3])])
                    j += 1
            
                ps1.append(firs)    
                c = len(firs)
#squeezing, spin length, difference between the last two best solutions, optimized parameters 
    return np.array(sq1), np.array(ln),np.array(dif), np.transpose(np.array(ps1), (0,2,1)) 
