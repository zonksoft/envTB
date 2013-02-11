Quantum capacitance
=======================

get_values_at_elements einbauen beim ladung fuer kapazität zusammenrechnen!

*Quantum capacitance* is the effect that the capacitance of a material with a 
small density of states at the Fermi energy is reduced. The effect is
temperature-dependent.

The code consists of two parts:

1. The function D(E,T) describes the density of states at a given energy E
   as a function of the temperature T. This is implemented for 
   :class:`graphene at 0K
   <quantumcapacitance.quantumcapacitance.FermiEnergyChargeDependence.bulk_graphene>` 
   (no temperature dependency, obviously) and :class:`graphene at finite 
   temperatures <quantumcapacitance.quantumcapacitance.BulkGrapheneWithTemperature>`.
2. The class :py:class:`.QuantumCapacitanceSolver` finds the electrostatic
   configuration that fulfils the constraints given by D(E,T).

The default usage is like this:

1. Create :py:class:`.QuantumCapacitanceSolver` object: 
   ``qcsolver=QuantumCapacitanceSolver(container,solver,elements,operator)``
2. Calculate classical dependency of charge on every possible potential 
   configuration of elements: ``qcsolver.refresh_basisvecs()``
3. Set potential/fermi_energy as you like (e.g. backgate, sidegates...).
4. Refresh the contribution of the new configuration:         
   ``qcsolver.refresh_environment_contrib()``
5. Solve: ``qcsolution=qcsolver.solve()``
6. go to 3) until you are through with all voltages you want to look at.

Example
-------

The function :py:func:`simple.GrapheneQuantumCapacitance` illustrates
the usage of the quantum capacitance solver::

	def GrapheneQuantumCapacitance():
	    """
	    Calculate the quantum capacitance of Graphene on SiO2, including temperature.
	    
	    Boundary conditions:
	
	    * left, right: periodic BC
	    * top: Neumann BC, slope=0
	    * bottom: backgate capacitor plate
	    
	    Please read the code comments for further documentation.
	    
	    """
	    #Set system parameters
	    gridsize=1e-9
	    hoehe=600
	    breite=1
	    backgatevoltage=0
	    graphenepos=299
	    temperature=300
	    sio2=3.9
	    
	    vstart=-60
	    vend=60
	    dv=0.5
	  
	    #Finite difference operator
	    lapl=electrostatics.Laplacian2D2ndOrderWithMaterials(gridsize,gridsize)
	    #Create Rectangle
	    periodicrect=electrostatics.Rectangle(hoehe,breite,1.,lapl)
	    
	    #Set boundary condition at the top
	    for y in range(breite):
	        periodicrect[0,y].neumannbc=(0,'xf')
	    
	    #Create list of backgate and GNR elements        
	    backgateelements=[periodicrect[hoehe-1,y] for y in range(breite)]
	    grapheneelements=[periodicrect[graphenepos,y] for y in range(breite)]
	    
	    #Set initial backgate element boundary condition (not necessary)
	    for element in backgateelements:
	        element.potential=backgatevoltage
	        
	    #Create D(E,T) function    
	    Ef_dependence_function=quantumcapacitance.BulkGrapheneWithTemperature(temperature,gridsize).Ef_interp
	    
	    #Set electrochemical potential and D(E,T) for the GNR elements
	    for element in grapheneelements:
	        element.potential=0
	        element.fermi_energy=0
	        element.fermi_energy_charge_dependence=Ef_dependence_function
	
	    #Set dielectric material
	    for x in range(graphenepos,hoehe):
	        for y in range(breite):
	            periodicrect[x,y].epsilon=sio2
	    
	    #Create periodic container    
	    percont=electrostatics.PeriodicContainer(periodicrect,'y')
	    
	    #Invert discretization matrix
	    solver,inhomogeneity=percont.lu_solver()
	    
	    #Create QuantumCapacitanceSolver object
	    qcsolver=quantumcapacitance.QuantumCapacitanceSolver(percont,solver,grapheneelements,lapl)
	    
	    #Refresh basisvectors for calculation. Necessary once at the beginning.
	    qcsolver.refresh_basisvecs()
	    
	    #Create volgate list
	    voltages=numpy.arange(vstart,vend,dv)
	    
	    charges=[]
	    
	    #Loop over voltages
	    for v in voltages:
	        #Set backgate elements to voltage
	        for elem in backgateelements:
	            elem.potential=v
	        #Check change of environment
	        qcsolver.refresh_environment_contrib()
	        #Solve quantum capacitance problem & set potential property of elements
	        qcsolution=qcsolver.solve()
	        #Create new inhomogeneity because potential has changed
	        inhom=percont.createinhomogeneity()
	        #Solve system with new inhomogeneity
	        sol=solver(inhom)
	        #Save the charge configuration in the GNR
	        charges.append(percont.charge(sol,lapl,grapheneelements))
	    #Sum over charge and take derivative = capacitance
	    totalcharge=numpy.array([sum(x) for x in charges])
	    capacitance=(totalcharge[2:]-totalcharge[:-2])/len(grapheneelements)*gridsize/(2*dv)
	
	    #Plot result
	    fig=pylab.figure()
	    ax = fig.add_subplot(111)
	    ax.set_title('Quantum capacitance of Graphene on SiO2')
	    ax.set_xlabel('Backgate voltage [V]')
	    ax.set_ylabel('GNR capacitance [$10^{-6} F/m^2$]')
	    pylab.plot(voltages[1:-1],1e6*capacitance)
    
The result: 

.. image:: image/quantumcapacitance_example.png

Code reference
--------------

.. autoclass:: quantumcapacitance.quantumcapacitance.BulkGrapheneWithTemperature
.. automethod:: quantumcapacitance.quantumcapacitance.FermiEnergyChargeDependence.bulk_graphene
.. autoclass:: quantumcapacitance.quantumcapacitance.QuantumCapacitanceSolver