import numpy as np
import copy as cp
from nptyping import NDArray, Float64, Int



class Vicsek_Class(object):
    """
    Description
    ---------------------------------------------------------------------------
    Implementation of modified Vicsek Model according to (I)Sumino et al. Nature (2012).
    Original Vicsek model was first described in (II) Vicsek et al. Phy Rev Let (1995).
    
    Vicsek_Class constructs a simulation box of size Lx, Ly on which the trajectories
    of N agents are evolved. The agents interact via a short-distance interaction
    potential. The radius of interaction is defined as r. To find the direct neighboring
    agents of a specific agent n_i Vicsek_class has been implemented with a fixed
    radius nearest neighbor search algorithm.
 
    

    Parameters
    ---------------------------------------------------------------------------
    N : int
        Number of agents
    Lx : float
        horizontal length of simulation box
    Ly : float
        vertical length of interaction box
    vel: float
        speed of the agents
    r: float
        interaction radius
    tau: float
        parameter for Ornstein-Uhlenbeck process defined in (I)eq.2
    alpha: float
        density weight - Parameter in (I) eq.3
    symmetry_parameter: float
        symmetry_parameter: either 1.0 for ferrromagnetic or 2.0 for nematic interaction
    diffusion_c: float
        parameter for Ornstein-Uhlenbeck process defined in (I)eq.2
    omega0: float
        parameter for Ornstein-Uhlenbeck process defined in (I)eq.2
    dt: float
        integration time step
    test: str
        if yes then each box has exactly one particle in the center of the box.
    
    
    Attributes
    ----------
    N : int
        Number of agents
    Lx : float
        horizontal length of simulation box
    Ly : float
        vertical length of interaction box
    v: float
        speed of the agents
    r: float
        interaction radius
    tau: float
        parameter for Ornstein-Uhlenbeck process defined in (I)eq.2
    alpha: float
        density weight - Parameter in (I) eq.3
    symmetry_parameter: float
        symmetry_parameter: either 1.0 for ferrromagnetic or 2.0 for nematic interaction
    diffusion: float
        parameter for Ornstein-Uhlenbeck process defined in (I)eq.2
    omega0:
        parameter for Ornstein-Uhlenbeck process defined in (I)eq.2
    dt:
        integration time step
    Nx: int
        floored horizontal size of box in units of r, i.e. floor(Lx/r)
    Ny: int
        floored Vertical of box in units of r, i.e. floor(Ly/r)
    box_size_x: float
        actual horizontal size of box in units of r, i.e. floor(Lx/r)
    box_size_y: float
        actual vertical size of box in units of r, i.e. floor(Lx/r)
    M: int
        Nx*Ny -> Number of squared sized boxed of length r covering the entire
        simulation box
    hash_table: dict
        keys: int
            range from 0 to M-1. Denotes the individual boxes inside the simulation 
            box.
        values: list
            indices of agents inside of box -> variable length
    keys: 1d numpy array of size N
        map agent to box_id/key in hash_table, i.e.
        agent <-> box_id
    hash_table_neighborboxes: dict
        create hash table where keys are box_ids and values are lists with 9 elements, 
        containing ids of the 8 neigbhbor boxes including the own.
    coord: 2d numpy array of size (N, 5)
        phase space information of system at time t_i
        coord ~ np.array([N rows (agents), 5 columns(space, direction, noise (from ornstein-uhlenbeck))])  
    coord_next: 2d numpy array
        coord at time step t_{i+1}
    order_parameter: float
        order parameter from (II) eq.2
    phi_array: 1d numpy array of size N
        Direction of movement of all agents. Used for animation purposes, not used in
        simulation.
    
   
    
    Examples
    --------
    Lx =3
    Ly = 3
    rho = 1
    N = 100
    # N = int(Lx*Ly*rho)
     
    r = 1.0
    dt = 1.0
     
    k0 = -7.1e-4
    s0 =  542
    alpha = 1 # density weight
    symmetry_parameter = 2 #(1 = ferromagnetic, 2 = nematic) allignment
    vel = 0.5
    omega0 = vel*k0
    tau = s0/vel # memory time
    sigma_k = 1.8e-3
    diffusion_c = np.sqrt(2*sigma_k**2*vel**3/s0)
    vicsek_obj = Vicsek_Class(N, Lx, Ly, vel, r, tau, alpha, symmetry_parameter, diffusion_c, omega0, dt)
    vicsek_obj.update()
    
    Sources
    --------
    http://efekarakus.github.io/fixed-radius-near-neighbor/
    
    """


    def __init__(self, N, Lx, Ly, vel, r, tau, alpha, symmetry_parameter, diffusion_c, omega0, dt):
    #constructor -> simulation parameters -> simulation variables"

        self.N = N #the number of the agents
        self.Lx = Lx #the size of the area the simulation is carried out
        self.Ly = Ly #the size of the area the simulation is carried out
        self.r = r #the interaction radius
        self.dt = dt # integration step

        self.v = vel #the speed of the agents
        self.tau = tau # memory time
        self.alpha = alpha # density weight
        self.symmetry_parameter = symmetry_parameter # ferrromagnetic or nematic
        self.diffusion = diffusion_c #OU process parameter
        self.omega0 = omega0 #OU process parameter

        self.Nx = int(self.Lx/self.r)
        self.Ny = int(self.Ly/self.r)
        self.box_size_x = self.Lx/float(int(self.Lx/self.r))
        self.box_size_y = self.Ly/float(int(self.Ly/self.r))
        self.M = self.Nx * self.Ny
        self.hash_table = {}
        self.get_hash_tables() #-hash_table, keys initalize
        self.get_hash_table_neighborboxes()
        # initialize simulation variables: coord, and coord_next.
        # coord = np.array([N rows (particles), 5 columns(space, direction, noise (from ornstein-uhlenbeck))])     
        self.coord = np.zeros((self.N, 5))
        self.coord[:,0] = self.Lx * np.random.uniform(0, 1, self.N)
        self.coord[:,1] = self.Ly * np.random.uniform(0, 1, self.N)
        

        #directions
        for l in range(0, self.N):
            self.coord[l, 2:4] = self.ang_to_vel(np.random.uniform(-np.pi, np.pi))

        #noise
        self.coord[:,4] = np.random.normal(loc=0.0, scale=1.0, size=self.N)

        #create update array of particles
        self.coord_next = cp.deepcopy(self.coord)

        #create initial order parameter
        self.order_parameter = 0
        self.update_order_parameter()

#        self.testvelocities()
        #class variable defined in the running simulation
        self.phi_array = np.array([])
    # -------------------------------------------------------------------------------------------------------
    #end of constructor -------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------


    #rotate vector by angle phi
    def _for_testing(self):
        self.coord[:, 0] = np.mod(np.arange(0, self.N, 1) + self.r/2, 3)
        self.coord[:, 1] = np.sort(np.mod(np.arange(0, self.N, 1) + self.r/2, 3))
        
    def ang_to_vel(self, phi: NDarray[Float64]) -> NDarray[Float64]:
        return self.v*np.cos(phi), self.v*np.sin(phi)
    # -------------------------------------------------------------------------------------------------------

    #return angle of vector
    def vel_to_ang(self, vy: NDarray[Float64], vx: NDarray[Float64]) -> NDarray[Float64]:
        return np.arctan2(vy,vx)
    # -------------------------------------------------------------------------------------------------------

    #get average angle from array of angles
    def get_avg(self, ang_array):
        x = np.sum(np.cos(ang_array))
        y = np.sum(np.sin(ang_array))
        return np.arctan2(y,x)
    # -------------------------------------------------------------------------------------------------------

    #move particles with euler step. also use modulo for periodic boundary conditions
    def move_agents(self):
        self.coord_next[:, 0] = np.mod(self.coord[:, 0] + self.coord[:, 2] * self.dt, self.Lx)
        self.coord_next[:, 1] = np.mod(self.coord[:, 1] + self.coord[:, 3] * self.dt, self.Ly)
    # -------------------------------------------------------------------------------------------------------

    #get array of all angles for all particles
    def get_angles(self):
        self.phi_array = (np.arctan2(self.coord[:,3], self.coord[:,2]))
    # -------------------------------------------------------------------------------------------------------

    #test velocities for normalization
    def testvelocities(self):
    	x = self.coord_next[:,2:4]
    	speed = np.sqrt(x[:,0]**2 + x[:,1]**2)
    	if np.any(~np.isclose(speed, self.v)):
    	    print("Velocites are not normalized, code is broken")
    # -------------------------------------------------------------------------------------------------------

    #initalize hash tables hash = {keys:value}, keys = box number (1d array index), value = list with ids of particles in box with number=key
    def get_hash_tables(self):
        #first check if Simulation box size and interaction radius are compatible.
        if (not np.isclose(self.Lx/self.r, int(self.Nx))):
            print("r and Nx or Ny not compatible, grid square size "+str(self.Nx)+"x"+str(self.box_size_x)+" - "+str(self.Ny)+"x"+str(self.box_size_y)+" greater than interaction radius"+str(self.r))
        box_id_arr = np.arange(0, self.M, dtype = np.int32) #array 0:M-1 of 1d box_ids.
        self.hash_table = {box_id:[] for box_id in box_id_arr} #create hash_table
        self.hash_table_neighborboxes = {box_id:[] for box_id in box_id_arr} #create hash_table of neighborboxes
        self.keys = np.zeros(self.N, dtype=np.int32) #construct keys array
    # -------------------------------------------------------------------------------------------------------

    #map agent_id onto key values in hash table and keys array
    def get_agent_ids_into_hash_table(self):
        for k in range(0, self.N): #loop over agents
            j = int(self.coord[k, 0]/self.r) #column id of box
            i = int(self.coord[k, 1]/self.r) #row id of box
            l = int(i * self.Nx + j)    #1d representation of box_id
            self.keys[k] = l #agent k is in box l
            self.hash_table[l].append(k) #append to list of agents already in l
    # -------------------------------------------------------------------------------------------------------

    #get keys of neighboring boxes for box number at grid position i,j
    def get_neighborboxes_ids(self, i, j):
        ne_boxes = [] #initalize list of neighbor boxes of box at indices i and j.
        for k in range(i - 1, i + 2): #loop over boxes one above and one below
            p = k % self.Ny #periodic boundary conditions
            for l in range(j - 1, j + 2): #loop over boxes one left and one right
                q = l % self.Nx #periodic boundary conditions
                z = int(p * self.Nx + q) #convert from 2d to 1d representation
                ne_boxes.append(z) #append 1d index of neighbor box
        return ne_boxes
    # -------------------------------------------------------------------------------------------------------

    #create hash table where keys are 1d box numbers and values are lists with 9 elements, containing ids of
    #the 8 neigbhbor boxes including the own
    def get_hash_table_neighborboxes(self):
        for i in range(0, self.M): #loop over all M boxes
            m, n = np.unravel_index(i, (int(self.Nx), int(self.Ny))) #2d representation of box position
            
            #box i has 9 neighbors, append to hash table of neighborboxes of i a list with 9 indices
            self.hash_table_neighborboxes[i].extend(self.get_neighborboxes_ids(m, n))
    # -------------------------------------------------------------------------------------------------------

    #shift position vectors if agents are identified in neighboring boxes due to 
    #periodic boundary conditions
    def shiftvectors(self, coord_ne, m, n, a, b):
        if  np.abs(m - a) > 1: #for y
            alpha = (m - a)/np.abs(m - a)
            coord_ne[:, 1] += alpha * self.Ly
        if np.abs(n - b) > 1: #for x
            beta = (n - b)/np.abs(n - b)
            coord_ne[:, 0] += beta * self.Lx
        return coord_ne
    
    
    # -------------------------------------------------------------------------------------------------------
    def get_neighbor_ind_i_nbor_id_norm(self, nbor_id, x_i, y_i, m, n):
        ind_in_box_ne = self.hash_table[nbor_id]   #id of agents in neigborbox ne
        coord_ne = self.coord[ind_in_box_ne, 0:2]         #slice coord array
        a, b = np.unravel_index(nbor_id, (int(self.Nx), int(self.Ny)))    #get 2d grid position of neighbor box ne
        if np.abs(m-a)>1 or np.abs(n-b)>1:                          #check for periodic boundary condition
            coord_ne = self.shiftvectors(coord_ne, m, n, a, b)      #account for periodic boundary condition
        norm_arr = np.sqrt((coord_ne[:, 0] - x_i)**2 + (coord_ne[:, 1] - y_i)**2)   #now calculate pure distance
        return norm_arr
    
    # -------------------------------------------------------------------------------------------------------
    #get ids of neighboring agent for agent i
    def get_neighbor_ind_i(self, i):
        neighbors_list = [] #initalize list of neighbor indices (1d representation)
        box_id = self.keys[i]  #agent i is in the box with id box_id
        m, n = np.unravel_index(box_id, (int(self.Nx), int(self.Ny)))  #get 2d grid positon m, n of box_id
        neighborboxes_id_list = self.hash_table_neighborboxes[box_id]   #neighborboxes of box of agent i
        x_i = self.coord[i, 0] #x coordinate of agent i
        y_i = self.coord[i, 1] #y coordinate of agent i
        
        for nbor_id in neighborboxes_id_list:    #loop over all 9 neighborboxes of agent i
            if not self.hash_table[nbor_id]:   #check if empty; if empty, then leave loop
                continue
            norm_arr = self.get_neighbor_ind_i_nbor_id_norm(nbor_id, x_i, y_i, m, n)            
            mask_arr = (norm_arr <= self.r)            #bool array, to check all elements of dist if they are less than r
            if np.any(mask_arr) == False:          #if particle i has no neighbors in this box with distance less than r, leave loop
                continue
            temp_arr = np.array(self.hash_table[nbor_id])    
            neighbors_ids = temp_arr[mask_arr].tolist()
            neighbors_list.append(neighbors_ids)         #append neighbors into neigbor list.

        neighbors_list = [x1 for x2 in neighbors_list for x1 in x2] #neighbors has following form [[], ... [],...]. we need 1d list.
        return neighbors_list
    # -------------------------------------------------------------------------------------------------------
     
    #update direction   
    def update_direction(self, i, neighbors):
        phi_i = self.vel_to_ang(self.coord[i, 3],self.coord[i, 2])  #angle of particle i
        phi_dot = 0     #initialize phi_dot

        for x in neighbors:
            phi_neighbor = self.vel_to_ang(self.coord[x, 3], self.coord[x, 2])
            phi_dot += np.sin(self.symmetry_parameter * (phi_neighbor - phi_i))
        
        phi_dot = phi_dot * self.alpha/len(neighbors) + self.coord[i, 4]
        phi_new = phi_i + phi_dot * self.dt
        self.coord_next[i, 2:4] = self.ang_to_vel(phi_new)
    # -------------------------------------------------------------------------------------------------------   
     
    #integrate ornstein-uhlenbeck process   
    def integrate_ou_process(self):
        drift = (self.coord[:, 4] - self.omega0)/self.tau
        noise = np.random.normal(loc=0.0, scale=1.0) * np.sqrt(self.dt)
        self.coord_next[:, 4] = self.coord[:, 4] + drift * self.dt + self.diffusion * noise
    # -------------------------------------------------------------------------------------------------------

    #update order parameter
    def update_order_parameter(self):
        self.sum_x = np.sum(self.coord_next[:,2])
        self.sum_y = np.sum(self.coord_next[:,3])
        self.order_parameter = 1./self.N * 1./self.v * np.sqrt((self.sum_x)**2+(self.sum_y)**2)
    # -------------------------------------------------------------------------------------------------------
    
    #run simulation, update system
    def update(self):
        #update steps
        self.get_agent_ids_into_hash_table()  # assign agents into boxes: fills hash_table, keys global
        self.get_angles()  #phi_array global
        self.move_agents()       #move agents
        self.integrate_ou_process() #integrate ornstein-uhlenbeck

        for i in range(0, self.N):
            neighborhood_ids = self.get_neighbor_ind_i(i)
            self.update_direction(i, neighborhood_ids)

        self.update_order_parameter() # update order parameter values
        # set coordiantes to new coordiantes
        self.coord = cp.deepcopy(self.coord_next)
#        self.testvelocities()
    # -------------------------------------------------------------------------------------------------------
    
    
#Lx =3
#Ly = 3
#rho = 1
#N = 9
# N = int(Lx*Ly*rho)

#r = 1.0
#dt = 1.0

#k0 = -7.1e-4
#s0 =  542
#alpha = 1 # density weight
#symmetry_parameter = 2 #(1 = ferromagnetic, 2 = nematic) allignment
#vel = 0.5
#omega0 = vel*k0
#tau = s0/vel # memory time
#sigma_k = 1.8e-3
#diffusion_c = np.sqrt(2*sigma_k**2*vel**3/s0)
#phaser = Vicsek_Class(N, Lx, Ly, vel, r, tau, alpha, symmetry_parameter, diffusion_c, omega0, dt)
#phaser.hash_table
#phaser.for_testing()
#phaser.get_agent_ids_into_hash_table()
# phaser.hash_table
# phaser.update()
#
#
#
# import matplotlib.pyplot as plt
# from matplotlib import collections  as mc 
# fig = plt.figure()
# ax1 = plt.subplot2grid((3,2),(1,0),colspan=2,rowspan=2)
# ax2 = plt.subplot2grid((3,2),(0,0),colspan=2)

    
# vvecs = phaser.coord[:,2:4]
# rvecs = phaser.coord[:,0:2]
# scat = ax1.scatter([], [], marker = ".")
# scat.set_offsets(rvecs)
# ax1.set_xlim(0, phaser.Lx)
# ax1.set_ylim(0, phaser.Ly)
# ax1.set_xlabel('Simulation')
# lines = []
# for i in range(0, phaser.N):
#     x = rvecs[i, 0]
#     y = rvecs[i, 1]
#     vx = vvecs[i, 0]
#     vy = vvecs[i, 1]
#     x1 = x + .5 * vx
#     y1 = y + .5 * vy
#     tp = [(x, y), (x1, y1)]
#     lines.append(tp)
# lc = mc.LineCollection(lines, colors = "r")
# ax1.add_collection(lc)

# ax2.set_xlim([0,10])
# ax2.set_ylim([0,1.1])
# ax2.set_title('Time')
# ax2.set_ylabel('$\phi$',fontsize=20)
# line, = ax2.plot([], [])        

# return scat, lc, line   

    
    

# Lx =3
# Ly = 3
# rho = 1
# N = 9
# r = 1.0
# dt = 1.0
# k0 = -7.1e-4
# s0 =  542
# alpha = 1 # density weight
# symmetry_parameter = 2 #(1 = ferromagnetic, 2 = nematic) allignment
# vel = 0.5
# omega0 = vel*k0
# tau = s0/vel # memory time
# sigma_k = 1.8e-3
# diffusion_c = np.sqrt(2*sigma_k**2*vel**3/s0)
# vicsek_obj = Vicsek_Class(N, Lx, Ly, vel, r, tau, alpha, symmetry_parameter, diffusion_c, omega0, dt)
