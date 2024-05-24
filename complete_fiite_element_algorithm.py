import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.core as cfc
import calfem.utils as cfu
import calfem.vis_mpl as cfv

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from sparse_assem import sp_assem

import numpy as np

from plantml import plantml


import math

# Set the element size and the type, 2 sets the type to traingular elements
el_type = 2

#create directory where mesh will be stored
mesh_dir = "./project_mesh"

#Create markers that will be tagged on the nodes that lie on some boundary
#The default marker 0 will be applied to all other nodes
MARKER_UPPER_BOUNDARY = 1
MARKER_LOWER_BOUNDARY = 2
MARKER_RED_CIRCLE = 3
MARKER_BLUE_CIRCLE = 4
MARKER_SYMMETRY_BOUNDARY = 5

#physical constants converted to SI units
c_p = 3600          #specific heat
rho = 540           #Density
k = 80              #thermal conductivity
alpha_c = 120       #Convection constant cooling cirles
alpha_n = 40        #Convection constant upper boundary
E = 5e9             #Youngs modulus
nu = 0.36           #Poisson ratio
alpha = 60e-6       #Expansion coefficient
t = 1.6             #Thickness of battery
G = E / (1 + nu)    #Shear modulus


#Define different temperatures

#the temperature outside the top boundary
T_inf = 293
#the initial temperature of the battery
T_0 = 293
#the temperature of the colling circle with red shading in the project description
T_out = 285
#the temperature of the cooling circle with blue shading in the project description
T_in = 277


time_step = 1 #Size of one time step 
number_of_time_steps = 1000 # number of time steps


def f1(t):
    return 100 * math.exp(-144*((600-t)/3600)**2) * 1e3
def f2(t):
    return 88.42 * (1 if t < 600 else 0) * 1e3

def generate_mesh(freedom_deg_per_node, show_geometry : bool, show_mesh : bool, el_size : int):
    #generate mesh
    g = cfg.geometry()
    
    #define parameteres
    L_LOWER = 0.300
    L_UPPER = 0.400
    HEIGTH = 0.200
    CIRCLE_RADIUS = 0.025
    DIST_BETWEEN_COOLING_HOLES = 0.0875
    
    #because of symmetry, only simulation of the right half of the battery is neccecary
    #define boundary points in clockwise direction
    g.point([0,0], 0)
    g.point([0, HEIGTH / 2 - CIRCLE_RADIUS], 1)
    g.point([CIRCLE_RADIUS, HEIGTH / 2], 2)
    g.point([0, HEIGTH/2 + CIRCLE_RADIUS], 3)
    g.point([0,HEIGTH], 4)
    g.point([L_UPPER, HEIGTH], 5)
    g.point([L_LOWER, 0], 6)

    #define cirlce centers
    g.point([0, HEIGTH / 2], 7)
    g.point([DIST_BETWEEN_COOLING_HOLES, HEIGTH / 2 ], 8)
    g.point([DIST_BETWEEN_COOLING_HOLES * 2, HEIGTH / 2], 9)
    g.point([DIST_BETWEEN_COOLING_HOLES * 3, HEIGTH / 2], 10)
    
    #define 4 points surrounding each circle centre in clockwise order
    #first point
    g.point([DIST_BETWEEN_COOLING_HOLES - CIRCLE_RADIUS, HEIGTH / 2], 11)
    g.point([DIST_BETWEEN_COOLING_HOLES,HEIGTH / 2 + CIRCLE_RADIUS], 12)
    g.point([DIST_BETWEEN_COOLING_HOLES + CIRCLE_RADIUS, HEIGTH / 2], 13)
    g.point([DIST_BETWEEN_COOLING_HOLES, HEIGTH/2 - CIRCLE_RADIUS], 14) 
    #second circle
    g.point([2 * DIST_BETWEEN_COOLING_HOLES - CIRCLE_RADIUS, HEIGTH / 2], 15)
    g.point([2*DIST_BETWEEN_COOLING_HOLES,HEIGTH / 2 + CIRCLE_RADIUS], 16)
    g.point([2*DIST_BETWEEN_COOLING_HOLES + CIRCLE_RADIUS, HEIGTH / 2], 17)
    g.point([2*DIST_BETWEEN_COOLING_HOLES, HEIGTH/2 - CIRCLE_RADIUS], 18) 
    #third circle
    g.point([3 * DIST_BETWEEN_COOLING_HOLES - CIRCLE_RADIUS, HEIGTH / 2], 19)
    g.point([3*DIST_BETWEEN_COOLING_HOLES,HEIGTH / 2 + CIRCLE_RADIUS], 20)
    g.point([3*DIST_BETWEEN_COOLING_HOLES + CIRCLE_RADIUS, HEIGTH / 2], 21)
    g.point([3*DIST_BETWEEN_COOLING_HOLES, HEIGTH/2 - CIRCLE_RADIUS], 22) 
    
    #define lines and circle segments
    g.spline([0,1], 0, marker = MARKER_SYMMETRY_BOUNDARY)
    g.circle([1, 7, 2], 1, marker = MARKER_RED_CIRCLE )
    g.circle([2, 7, 3], 2, marker= MARKER_RED_CIRCLE)
    g.spline([3, 4], 3, marker = MARKER_SYMMETRY_BOUNDARY)
    g.spline([4,5], 4, marker = MARKER_UPPER_BOUNDARY)
    g.spline([5, 6], 5, marker= MARKER_LOWER_BOUNDARY)
    g.spline([6, 0], 6, marker = MARKER_LOWER_BOUNDARY)
    
    #define full circle
    g.circle([11, 8, 12], 7, marker=MARKER_BLUE_CIRCLE)
    g.circle([12, 8, 13], 8, marker=MARKER_BLUE_CIRCLE)
    g.circle([13, 8, 14], 9, marker=MARKER_BLUE_CIRCLE)
    g.circle([14, 8, 11], 10, marker=MARKER_BLUE_CIRCLE)    

    #define second circle
    g.circle([15, 9, 16], 11, marker=MARKER_RED_CIRCLE)
    g.circle([16, 9, 17], 12, marker=MARKER_RED_CIRCLE)
    g.circle([17, 9, 18], 13, marker=MARKER_RED_CIRCLE)
    g.circle([18, 9, 15], 14, marker=MARKER_RED_CIRCLE)  
    
    #define third circle
    g.circle([19, 10, 20], 15, marker=MARKER_BLUE_CIRCLE)
    g.circle([20, 10, 21], 16, marker=MARKER_BLUE_CIRCLE)
    g.circle([21, 10, 22], 17, marker=MARKER_BLUE_CIRCLE)
    g.circle([22, 10, 19], 18, marker=MARKER_BLUE_CIRCLE) 
    
    #define which lines, circles and circlesegments are in the simulation
    g.surface([0,1,2,3, 4, 5, 6], holes=[[7, 8, 9, 10], [11, 12, 13, 14], [15,16,17,18]])

    
    #generate mesh
    mesh = cfm.GmshMeshGenerator(g, mesh_dir=mesh_dir)
    #set size of individual mesh elements
    mesh.el_size_factor = el_size
    #set element type, 2 is triangle 
    mesh.el_type = el_type
    #set degrees of freedom per mesh element
    mesh.dofs_per_node = freedom_deg_per_node   
    

    coord, edof, dofs, bdofs, element_markers = mesh.create()
    
        # display mesh
    if show_geometry:
        fig, ax = plt.subplots()
        cfv.draw_geometry(
            g,
            label_curves=True,
            title="Geometry: Computer Lab Exercise 2"
        )
        plt.show()
    
    if show_mesh:
        cfv.figure()
        cfv.drawMesh(
            coords=coord,
            edof = edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title = 'mesh'
        )
        boundary_nodes = bdofs[MARKER_UPPER_BOUNDARY]
        bx, by = [], []
        for node in boundary_nodes:
            bx.append(coord[node-1][0])
            by.append(coord[node-1][1])

        # Plotting the boundary nodes on the mesh
        plt.scatter(bx, by, c='red', s=50, label='Upper Boundary Nodes')
        
        # Add a legend to help identify the overlay
        plt.legend()

        # Display the plot
        cfv.showAndWait()
        
        cfv.figure()
        cfv.drawMesh(
            coords=coord,
            edof = edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title = 'mesh'
        )
        boundary_nodes = bdofs[MARKER_LOWER_BOUNDARY]
        bx, by = [], []
        for node in boundary_nodes:
            bx.append(coord[node-1][0])
            by.append(coord[node-1][1])

        # Plotting the boundary nodes on the mesh
        plt.scatter(bx, by, c='red', s=50, label='Lower boundary')
        
        # Add a legend to help identify the overlay
        plt.legend()

        # Display the plot
        cfv.showAndWait()
        
        cfv.figure()
        cfv.drawMesh(
            coords=coord,
            edof = edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title = 'mesh'
        )
        boundary_nodes = bdofs[MARKER_RED_CIRCLE]
        bx, by = [], []
        for node in boundary_nodes:
            bx.append(coord[node-1][0])
            by.append(coord[node-1][1])

        # Plotting the boundary nodes on the mesh
        plt.scatter(bx, by, c='red', s=50, label='Red circle')
        
        # Add a legend to help identify the overlay
        plt.legend()

        # Display the plot
        cfv.showAndWait()
        
        cfv.figure()
        cfv.drawMesh(
            coords=coord,
            edof = edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title = 'mesh'
        )
        boundary_nodes = bdofs[MARKER_BLUE_CIRCLE]
        bx, by = [], []
        for node in boundary_nodes:
            bx.append(coord[node-1][0])
            by.append(coord[node-1][1])

        # Plotting the boundary nodes on the mesh
        plt.scatter(bx, by, c='red', s=50, label='Blue circle')
        
        # Add a legend to help identify the overlay
        plt.legend()

        # Display the plot
        cfv.showAndWait()
        
        cfv.figure()
        cfv.drawMesh(
            coords=coord,
            edof = edof,
            dofs_per_node=mesh.dofsPerNode,
            el_type=mesh.elType,
            filled=True,
            title = 'mesh'
        )
        boundary_nodes = bdofs[MARKER_SYMMETRY_BOUNDARY]
        bx, by = [], []
        for node in boundary_nodes:
            bx.append(coord[node-1][0])
            by.append(coord[node-1][1])

        # Plotting the boundary nodes on the mesh
        plt.scatter(bx, by, c='red', s=50, label='Symmetry')
        
        # Add a legend to help identify the overlay
        plt.legend()

        # Display the plot
        cfv.showAndWait()
        
    
        
        
    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    print(dofs.shape)
    print(edof.shape)
    
    return (coord, edof, dofs, bdofs, bc, bc_value, element_markers)

def calculate_temperature_distribution(coord, edof, dofs, bdofs):
    # Define the thermal conductivity matrix D
    D = np.array([
        [k, 0],
        [0, k]
    ])

    # Initialize global stiffness matrix K
    K = sp.lil_matrix((dofs.size, dofs.size))

    # Initialize global force vector due to convection
    F_c = np.zeros((dofs.size, 1))
    # Initialize global C matrix
    C = sp.lil_matrix((dofs.size, dofs.size))
    # Initialize global load vector
    F_l = np.zeros((dofs.size, 1))

    # Local stiffness matrix configurations for element connectivity
    K_ce12 = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 0]])
    K_ce13 = np.array([[2, 0, 1], [0, 0, 0], [1, 0, 2]])
    K_ce23 = np.array([[0, 0, 0], [0, 2, 1], [0, 1, 2]])

    # Local force vector configurations for thermal convection boundary
    f_ce12 = np.array([[1/2], [1/2], [0]])
    f_ce13 = np.array([[1/2], [0], [1/2]])
    f_ce23 = np.array([[0], [1/2], [1/2]])
    
    upper_nodes_found_x, upper_nodes_found_y = [], []
    

    # Iterate over each element defined in edof
    for element_x in range(edof.shape[0]):
        # Extract node coordinates for the current element
        ex, ey = coord[edof[element_x, :] - 1, :].T
        
        # Calculate local stiffness matrix K_e 
        K_e = cfc.flw2te(ex, ey, ep=[t], D=D)

        # Extract nodes for the current element
        nodes = edof[element_x]

        # Initialize local force vector and stiffness matrix for boundary conditions
        f_ce = np.zeros((3, 1))
        K_ce = np.zeros((3, 3))

        # Lists to track nodes on different boundaries
        upper_boundary = []
        red_circle_boundary = []
        blue_circle_boundary = []

        # Determine which nodes are on specific boundaries
        for n in range(3):
            if nodes[n] in bdofs[MARKER_UPPER_BOUNDARY]:
                upper_boundary.append(n)
            if nodes[n] in bdofs[MARKER_RED_CIRCLE]:
                red_circle_boundary.append(n)
            if nodes[n] in bdofs[MARKER_BLUE_CIRCLE]:
                blue_circle_boundary.append(n)
        
        # Calculate element area using determinant of the coordinate matrix
        Cmat = np.vstack((np.ones((3,)), ex, ey))
        area = np.linalg.det(Cmat) / 2

        # Compute local load vector f_le due to distributed sources
        f_le = np.ones((3, 1)) * area * t / 3

        # Apply boundary conditions based on detected boundary nodes
        if len(upper_boundary) == 2:
            # Calculate length of the boundary edge
            l = ((ex[upper_boundary[0]] - ex[upper_boundary[1]])**2 
                 + (ey[upper_boundary[0]] - ey[upper_boundary[1]])**2)**0.5
            
            upper_nodes_found_x.append(ex[upper_boundary[0]])
            upper_nodes_found_x.append(ex[upper_boundary[1]])
            
            upper_nodes_found_y.append(ey[upper_boundary[0]])
            upper_nodes_found_y.append(ey[upper_boundary[1]])
            # Select appropriate stiffness matrix and force vector for boundary condition
            if 0 in upper_boundary and 1 in upper_boundary:
                K_ce += K_ce12
                f_ce += f_ce12
            elif 0 in upper_boundary and 2 in upper_boundary:
                K_ce += K_ce13
                f_ce += f_ce13
            elif 1 in upper_boundary and 2 in upper_boundary:
                K_ce += K_ce23
                f_ce += f_ce23

            # Scale boundary condition matrices by properties calulated in theory
            K_ce *= alpha_n * l / 6 * t
            f_ce *= alpha_n * T_inf * l * t

        elif len(red_circle_boundary) == 2:
            # Calculate length of the boundary edge
            l =  ((ex[red_circle_boundary[0]] - ex[red_circle_boundary[1]])**2 
                  + (ey[red_circle_boundary[0]] - ey[red_circle_boundary[1]])**2)**0.5

            # Select appropriate stiffness matrix and force vector for boundary condition
            if 0 in red_circle_boundary and 1 in red_circle_boundary:
                K_ce += K_ce12
                f_ce += f_ce12

            elif 0 in red_circle_boundary and 2 in red_circle_boundary:
                K_ce += K_ce13
                f_ce += f_ce13

            elif 1 in red_circle_boundary and 2 in red_circle_boundary:
                K_ce += K_ce23
                f_ce += f_ce23

            # Scale boundary condition matrices by properties calculated in theory
            K_ce *= alpha_c* l/6 * t
            f_ce *= alpha_c * T_out * l * t

        elif len(blue_circle_boundary) == 2:
            # Calculate length of the boundary edge
            l =  ((ex[blue_circle_boundary[0]] - ex[blue_circle_boundary[1]])**2 
                  + (ey[blue_circle_boundary[0]] - ey[blue_circle_boundary[1]])**2)**0.5
          
            # Select appropriate stiffness matrix and force vector for boundary condition
            if 0 in blue_circle_boundary and 1 in blue_circle_boundary:
                K_ce += K_ce12
                f_ce += f_ce12
            elif 0 in blue_circle_boundary and 2 in blue_circle_boundary:
                K_ce += K_ce13
                f_ce += f_ce13
            elif 1 in blue_circle_boundary and 2 in blue_circle_boundary:
                K_ce += K_ce23
                f_ce += f_ce23
                
            # Scale boundary condition matrices by properties calculated in theory
            K_ce *= alpha_c*l/6*t
            f_ce *= alpha_c * T_in * l * t

        #compute the C matrix using materieal properties
        C_e = plantml(ex, ey, rho * c_p * t)

        #assemble local element matricies into global matricies
        cfc.assem(edof[element_x, :], K, K_e + K_ce, F_c, f_ce)  
        cfc.assem(edof[element_x, :], C, C_e, F_l, f_le)
    
    #initial solution matrix a with initial temperature distribution T_0
    #each column represents the nodal temperatures at time step t
    a = np.zeros((dofs.size, number_of_time_steps))
    a[:, 0] = T_0
    
    #define load function used for transient analysis
    load_function = lambda t: f2(t)
    
    # choose integration method
    # 1 - implicit 
    # 1/2 crank-nickolson
    theta = 1
    
    C = sp.csr_matrix(C)
    K = sp.csr_matrix(K)


    #time stepping loop to solve for temperature at each time step
    for i in range(1, number_of_time_steps):
        
        
        left_side = C * 1 / time_step + K*theta
        f_prev = F_c + load_function(time_step*(i - 1))*F_l
        f_next = F_c + load_function(time_step*i)*F_l       
        
        a_prev = a[:, i - 1].reshape((dofs.size, 1))
        
        right_side = (theta * f_next + (1 - theta) * f_prev + C.dot(a_prev) 
                      * (1 / time_step) - K.dot(a_prev) * (1 - theta))

        # Solve the linear system using a sparse solver
        a_next = splinalg.spsolve(left_side, right_side)
        
        # Update the temperature matrix for the current time step
        a[:, i] = a_next

        # Print progress
        if (i + 1) % 100 == 0:
            print(f'Time Step {i + 1}/{number_of_time_steps} Complete')
    
    #return matrix detrcibing all nodal temperatures at all time steps
    return a, upper_nodes_found_x, upper_nodes_found_y

def visualize_temperature_distribution(temperatures, coord, edof):
    plt.ion()
    cfv.figure()
    for i in range(0, temperatures.shape[1], 50):
        cfv.draw_nodal_values(temperatures[:, i], coord, edof, levels=50, 
                              title=f"Temperature Distribution t={i*time_step}")
        cfv.colorbar()
        plt.show()
        plt.pause(0.01)
        plt.clf()
        
def visualize_max_temp(temp,cord,edof, time):
    cfv.figure()
    cfv.draw_nodal_values(T_0 + temp, coord, edof, levels=100, 
                            title=f"Temperature Distribution at t = {time} with max temp = {round(max(T_0 + temp),2)}")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    cfv.colorbar()
    plt.show()
    plt.pause(0.01)
    plt.clf()
    
    
    

def plot_temperatures(temperatures):
    times = [time_step*i for i in range(number_of_time_steps)]

    min_temperatures = [min(temperatures[:, i]) for i in range(temperatures.shape[1])]
    max_temperatures = [max(temperatures[:, i]) for i in range(temperatures.shape[1])]
    max_deviations = [max(abs(max_temperatures[i] - T_0), abs(min_temperatures[i] - T_0)) 
                      for i in range(temperatures.shape[1])]

    plt.figure()

    plt.plot(times, min_temperatures, label='Min temperature', color='blue')
    plt.plot(times, max_temperatures, label='Max temperature', color='red')
    # plt.plot(times, max_deviations, label='Max deviation from initial', color='green')

    # Add labels and legend
    plt.xlabel('Time')
    plt.ylabel('Temperature (K)')
    plt.legend()

    plt.show()

def find_temperatures_at_max_deviation(edof, temperatures):
    # Calculate the minimum temperature for each column in the temperature matrix
    min_temperatures = [min(temperatures[:, i]) for i in range(temperatures.shape[1])]
    # Calculate the maximum temperature for each column in the temperature matrix
    max_temperatures = [max(temperatures[:, i]) for i in range(temperatures.shape[1])]
    # Calculate the maximum deviation from the initial temperature
    max_deviations = [max(abs(max_temperatures[i] - T_0), abs(min_temperatures[i] - T_0)) 
                      for i in range(temperatures.shape[1])]

    # Find the maximum value among the calculated deviations
    max_deviation = max(max_deviations)
    # Get the index of the column with the maximum deviation
    max_deviation_idx = max_deviations.index(max_deviation)
    # Calculate absolute deviations from T_0 for the temperatures in the column with the maximum deviation
    deviations = abs(T_0 - temperatures[:, max_deviation_idx])
    # Initialize an array for storing element-wise average deviations
    element_wise_deviations = np.zeros((edof.shape[0]))

    # Calculate element-wise average deviations
    for i, nodes in enumerate(edof):
        # For each element defined by nodes, calculate the average deviation
        element_wise_deviations[i] = sum([deviations[j - 1] for j in nodes]) / 3

    # Return the array of element-wise deviations
    return element_wise_deviations, deviations, max_deviation_idx


def calculate_stress_distribution(coord, edof, dofs, bdofs, temperature_deviations, ux, uy):
    
    #extract x and y coordinates for every element. 
    ex, ey = cfc.coordxtr(edof, coord, dofs)

    #define our matricies
    K = sp.lil_matrix((dofs.size, dofs.size))
    f_e = np.zeros((dofs.size, 1))
    ep = [2, t]
    
    #we chose to define a 3x3 D matrix and then calculate out of the plane stress separately
    D = np.array([ [1 - nu, nu, 0],
                        [nu, 1 - nu, 0],
                        [0, 0, 0.5 * (1 - 2 * nu)]])

    D *= (E / ((1 + nu) * (1 - 2*nu)))
    
    bc, bc_value = np.array([], 'i'), np.array([], 'f')
    
    #setting 0 displacement along the lower and right bounadry. 
    # Setting 0 displacement in the x direction along the symmetry axis
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_LOWER_BOUNDARY,0, 0)
    bc, bc_value = cfu.applybc(bdofs, bc, bc_value, MARKER_SYMMETRY_BOUNDARY, 0, 1)


    for i, (nodes, element_x, element_y) in enumerate(zip(edof, ex, ey)):
        #Create elementwise K
        K_e = cfc.plante(element_x, element_y, ep, D)

        #calculate D epsilon_0
        es = np.matmul(D, np.array([[1], [1], [0]])) *(1 + nu) *  alpha * temperature_deviations[i]

        #calculate force vector
        f_ee = cfc.plantf(element_x, element_y, ep, es.T).reshape((6, 1))

        #assemble
        cfc.assem(nodes, K, K_e, f_e, f_ee)

    #converte the matricies to sparse matricies for more efficient computation
    K = sp.csr_matrix(K)
    f_e = sp.csr_matrix(f_e)
    
    a, r = cfc.spsolveq(K, f_e, bc, bc_value)
    
    #extract the displacement for each element
    ed = cfc.extract_eldisp(edof,a)

    von_mises = []

    # Find the von-mises stress for each mesh element
    node_stresses = [[] for _ in range(dofs.shape[0])]
    for i in range(edof.shape[0]):
        # Determine element stresses and strains in the element.
        es, et = cfc.plants(ex[i,:], ey[i,:], [2,t], D, ed[i,:])
        
        #sice our D matrix is 3x3 this is our output 
        sig_x, sig_y, tau_xy = es.T
        
        const = (alpha * E * temperature_deviations[i]) / (1 - 2 * nu)
    
        sig_x = sig_x[0] - const
        sig_y = sig_y[0] - const
        sig_z = nu * (sig_x + sig_y) - alpha * E * temperature_deviations[i] #add out of he plane behaviour
        tau_xy = tau_xy[0]
        tau_yz, tau_xz = 0, 0 #set these to 0 because these forces do not exist in this model of the problem
    
        #calculate the von-mises stress of the element 
        von_mises_element_stress = math.sqrt(sig_x**2+sig_y**2+sig_z**2 - sig_x*sig_y - sig_x*sig_z 
                                             - sig_y*sig_z + 3*tau_xy**2 + 3*tau_xz**2 + 3*tau_yz**2)
        von_mises.append(von_mises_element_stress)
        
        #apply this to all nodes that are in contact with this element
        #the nodal element is the x degree of freedom floor divided by 2
        x1, x2, x3 = edof[i, 0], edof[i, 2], edof[i, 4]
        node_stresses[x1 // 2].append(von_mises_element_stress)
        node_stresses[x2 // 2].append(von_mises_element_stress)
        node_stresses[x3 // 2].append(von_mises_element_stress)
        
        
    #set the nodal stress to the mean of all contributions to it
    avg_node_stresses = [np.mean(node_stresses[i]) for i in range(dofs.shape[0])]
    
    
    print("maximal von mises stress", max(von_mises))
    
    return von_mises, avg_node_stresses, a
    

    

    
    
def draw_elementwise_stress(von_mises, coord, edof):
    cfv.figure(fig_size=(10,10))
    cfv.draw_element_values(von_mises, coord, edof, 2 ,2, None, draw_elements=False, 
                            draw_undisplaced_mesh=False, title=f"Effective stress with {edof.shape[0]} elements, max stress = {round(max(von_mises)/1000000,2)}MPa")
    cfv.colorbar()

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    cfv.show_and_wait()
    
def draw_displacement(displacement, coord, edof):
    cfv.figure(fig_size=(10,10))
    cfv.draw_displacements(100 * displacement, coord, edof, 2, 2, 
                           draw_undisplaced_mesh=True, title="Displacements")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    cfv.show_and_wait()
    
def draw_nodal_stresses(avg_node_stresses, coord, edof):
    cfv.figure()  # Start a new figure
    cfv.draw_nodal_values(avg_node_stresses, coord, edof, levels=200, 
                          title=f"Nodal stress distribution with {edof.shape[0]} elements, max stress = {round(max(avg_node_stresses)/1000000,2)}MPa")
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    cfv.colorbar()  # Add a colorbar to the figure to indicate temperature scale
    cfv.showAndWait()  # Display the plot and wait until it is closed

    
def dofs_edofs_expandion(dofs, edofs, bdofs):
    
    dofs2 = [[0,0] for _ in range(dofs.size)]
    id = 1
    for row in range(dofs.size):
        dofs2[row][0] = id
        id += 1 
        dofs2[row][1] = id
        id+=1
        
    #create new edofs
    edof2 = [[0, 0, 0, 0, 0, 0] for _ in range(edof.shape[0])]
    
    new_bdofs = {}
    for k in bdofs.keys():
        new_bdofs[k] = []
        
    """
    expansion algorithm:
    every entry in the original edof matrix, corresponds to a row element in the original dofs matrix
    every integer in the original edof, will map to a row in the origal dofs
    this will map to the same row idx in the new dofs.
    The new row in dofs, has two entries, where none of them are the same as the original integer. 
    This integer is then replaced by this integer pair as the new degrees of freedom
    """
    for row in range(edof.shape[0]):
        r_id_1, r_id_2, r_id_3 = edof[row, :]
        

        
        f1, f2 = dofs2[r_id_1 - 1][0], dofs2[r_id_1 - 1][1]
        edof2[row][0] = f1
        edof2[row][1] = f2
        
        for k in bdofs.keys():
            if r_id_1 in bdofs[k]:
                new_bdofs[k].append(f1)
                new_bdofs[k].append(f2)
        
        f3, f4 = dofs2[r_id_2 - 1][0], dofs2[r_id_2 - 1][1]
        edof2[row][2] = f3 
        edof2[row][3] = f4
        
        for k in bdofs.keys():
            if r_id_2 in bdofs[k]:
                new_bdofs[k].append(f3)
                new_bdofs[k].append(f4)
        
        f5, f6 = dofs2[r_id_3 - 1][0], dofs2[r_id_3 - 1][1]
        edof2[row][4] = f5
        edof2[row][5] = f6
        
        for k in bdofs.keys():
            if r_id_3 in bdofs[k]:
                new_bdofs[k].append(f5)
                new_bdofs[k].append(f6)
        
    return np.array(dofs2), np.array(edof2), new_bdofs
        
    
if __name__=="__main__":
    elements = []
    stress = []
    for sz in [0.5, 0.2, 0.1, 0.08, 0.065, 0.05,0.035, 0.025, 0.02, 0.0175, 0.015,0.0125, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]:

        coord, edof, dofs, bdofs, bc, bc_value, element_markers = generate_mesh(1, False, False, sz )
        temp, ux, uy = calculate_temperature_distribution(coord, edof, dofs, bdofs)

        max_deviations, nodal_deviations, time = find_temperatures_at_max_deviation(edof, temp)


        final_dofs, final_edof, final_bdofs = dofs_edofs_expandion(dofs, edof, bdofs)

        von_mis, avg_von_mis, displacement = calculate_stress_distribution(coord, final_edof, final_dofs, 
                                                                        final_bdofs, max_deviations, ux, uy)

        draw_elementwise_stress(von_mis, coord, final_edof)
        draw_nodal_stresses(avg_von_mis, coord, edof)
        
        elements.append(edof.shape)
        stress.append(max(von_mis))
        print(elements)
        print(stress)
    
    
    print(elements)
    print(stress)
    visualize_max_temp(nodal_deviations, coord, edof, time)
    #plot_temperatures(temp)
    #visualize_temperature_distribution(temp, coord, edof)
    
    #draw_elementwise_stress(von_mis, coord, final_edof)
    #draw_nodal_stresses(avg_von_mis, coord, edof) 
    
    #draw_displacement(displacement, coord, final_edof)
