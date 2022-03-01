# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 00:13:09 2022

@author: Belal
"""

#import sys
#sys.path.append("../src/")
from Vicsek_Class import Vicsek_Class
import numpy as np
import unittest






class TestVicsek(unittest.TestCase):

    def setUp(self):
        self.Lx =3
        self.Ly = 3
        rho = 1
        self.N = 9
        self.r = 1.0
        self.dt = 1.0
        self.k0 = -7.1e-4
        self.s0 =  542
        self.alpha = 1 # density weight
        self.symmetry_parameter = 2 #(1 = ferromagnetic, 2 = nematic) allignment
        self.vel = 0.5
        self.omega0 = self.vel*self.k0
        self.tau = self.s0/self.vel # memory time
        self.sigma_k = 1.8e-3
        self.diffusion_c = np.sqrt(2*self.sigma_k**2 * self.vel**3/self.s0)
        self.vicsek_obj = Vicsek_Class(self.N, self.Lx, self.Ly, self.vel, self.r,
                 self.tau, self.alpha, self.symmetry_parameter, self.diffusion_c, 
                    self.omega0, self.dt)
        self.vicsek_obj._for_testing()
        self.vicsek_obj.update()
        self.vicsek_obj._for_testing()
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
        #check that each box has exactly one agent after applying ._for_testing()
        hash_table_values_list = list(self.vicsek_obj.hash_table.values())
        hash_table_values_arr = np.array([item for sublist in hash_table_values_list for item in sublist])
        what_we_expect = np.arange(0, self.N, 1)
        self.assertTrue(np.array_equal(hash_table_values_arr, what_we_expect))

    def test_get_hash_table_neighborboxes(self):
        nbors_8_list = self.vicsek_obj.hash_table_neighborboxes[8]
        nbors_8_list.sort()
        what_we_expect =  [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.assertTrue(nbors_8_list , what_we_expect)
    def test_get_neighborboxes_ids(self):
        ne_boxes_list_box_ind_0 = self.vicsek_obj.get_neighborboxes_ids(0, 0)
        what_we_expect = [8, 6, 7, 2, 0, 1, 5, 3, 4]
        self.assertTrue(ne_boxes_list_box_ind_0, what_we_expect)
    
    def test_shiftvectors(self):
        coord_5 = self.vicsek_obj.coord[[4], :]
        a, b = (1, 2) #2d indices of box 5
        m, n = (1, 0) #2d indices of box 3
        coord_new = self.vicsek_obj.shiftvectors(coord_5, m, n, a, b)
        # print(coord_new[0,:2])
        what_we_expect = np.array([-1.5, 1.5])
        self.assertTrue(np.array_equal(coord_new[0,:2], what_we_expect))
    
    
    def get_neighbor_ind_i(self):
        neighbors_0 = self.vicsek_obj.get_neighbor_ind_i(0)
        neighbors_0.sort()
        what_we_expect = [0, 1, 2, 3, 6]
        self.assertTrue(neighbors_0, what_we_expect)

if __name__ == '__main__':

    unittest.main()
