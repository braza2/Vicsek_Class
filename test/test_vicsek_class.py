# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 00:13:09 2022

@author: Belal
"""
from Vicsek_Class import Vicsek_Class
import numpy as np
import unittest






class TestVicsek(unittest.TestCase):
# we test the scenario of 9 particles in a box of size 3x3. Each particle sits in the
#middle of one of the 9 boxes. Please remember the periodic boundary conditions.

    """
       column     0 1 2
    row        8 |6 7 8| 6
    -------------|-----|----
    0          2 |0 1 2| 0
    1          5 |3 4 5| 3
    2          8 |6 7 8| 6
    -------------|-----|----
               2 |0 1 2| 0
    """
    def  test_get_agent_ids_into_hash_table(self):
        #check that each box has exactly one agent after applying .for_testing()
        hash_table_values_list = list(vicsek_obj.hash_table.values())
        hash_table_values_arr = np.array([item for sublist in hash_table_values_list for item in sublist])
        what_we_expect = np.arange(0, N, 1)
        self.assertTrue(np.array_equal(hash_table_values_arr, what_we_expect))

    def test_get_hash_table_neighborboxes(self):
        nbors_8_list = vicsek_obj.hash_table_neighborboxes[8]
        nbors_8_list.sort()
        what_we_expect =  [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.assertTrue(nbors_8_list , what_we_expect)
    def test_get_neighborboxes_ids(self):
        ne_boxes_list_box_ind_0 = vicsek_obj.get_neighborboxes_ids(0, 0)
        what_we_expect = [8, 6, 7, 2, 0, 1, 5, 3, 4]
        self.assertTrue(ne_boxes_list_box_ind_0, what_we_expect)
    
    def test_shiftvectors(self):
        coord_5 = vicsek_obj.coord[[4], :]
        a, b = (1, 2) #2d indices of box 5
        m, n = (1, 0) #2d indices of box 3
        coord_new = vicsek_obj.shiftvectors(coord_5, m, n, a, b)
        # print(coord_new[0,:2])
        what_we_expect = np.array([-1.5, 1.5])
        self.assertTrue(np.array_equal(coord_new[0,:2], what_we_expect))
    
    
    def get_neighbor_ind_i(self):
        neighbors_0 = vicsek_obj.get_neighbor_ind_i(0)
        neighbors_0.sort()
        what_we_expect = [0, 1, 2, 3, 6]
        self.assertTrue(neighbors_0, what_we_expect)

if __name__ == '__main__':
    Lx =3
    Ly = 3
    rho = 1
    N = 9
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
    vicsek_obj.for_testing()
    vicsek_obj.update()
    vicsek_obj.for_testing()
    unittest.main()