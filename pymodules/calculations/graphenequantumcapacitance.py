from quantumcapacitance import electrostatics,quantumcapacitance
from quantumcapacitance.common import Constants
import numpy

def PeriodicGraphenePatchWithGrapheneSidegatesSiO2Rectangle(breite,hoehe,temperature,graphenepos,graphenebreite=None,sidegatebreite=None):
    """
    Default potential for backgate + Graphene parts is 0V. If you want it differently, set it later using the returned references.
    """
    if (graphenebreite==None):
        graphenebreite=breite
        
    if (sidegatebreite==None):
        sidegatebreite=0        
        
    sio2=3.9
    gridsize=1e-9
    lapl=electrostatics.Laplacian2D2ndOrderWithMaterials(gridsize,gridsize)
    periodicrect=electrostatics.Rectangle(hoehe,breite,1.,lapl)
    for y in range(breite):
        periodicrect[0,y].neumannbc=(0,'xf')
        
    backgateelements=[periodicrect[hoehe-1,y] for y in range(breite)]
    
    for element in backgateelements:
        element.potential=0
        
    grapheneelements=[periodicrect[graphenepos,y] for y in range((breite-graphenebreite)/2,(breite+graphenebreite)/2)]
    
    Ef_dependence_function=quantumcapacitance.BulkGrapheneWithTemperature(temperature,gridsize).Ef_interp
    for element in grapheneelements:
        element.potential=0
        element.fermi_energy=0
        #periodicrect[graphenepos,y].fermi_energy_charge_dependence=FermiEnergyChargeDependence.bulk_graphene
        element.fermi_energy_charge_dependence=Ef_dependence_function
        
    graphenesidegateelementsleft=[periodicrect[graphenepos,y] for y in range(sidegatebreite)]
    graphenesidegateelementsright=[periodicrect[graphenepos,y] for y in range(breite-sidegatebreite,breite)]    
    for element in graphenesidegateelementsleft+graphenesidegateelementsright:
        element.potential=0
        #periodicrect[graphenepos,y].fermi_energy_charge_dependence=FermiEnergyChargeDependence.bulk_graphene
        element.fermi_energy_charge_dependence=Ef_dependence_function
    
    
    for x in range(graphenepos,hoehe):
        for y in range(breite):
            periodicrect[x,y].epsilon=sio2
    
    return periodicrect,lapl,grapheneelements,graphenesidegateelementsleft,graphenesidegateelementsright,backgateelements

def QuantumCapacitanceVoltageSweep(qcsolver,container,charge_operator,solver,voltages,voltage_elements,charge_elements): #all elements whose charge shall be saved are in charge_elements, i.e. graphene, sidegate etc.
    charges=[]
    for v in voltages:
        print v
        for elem in voltage_elements:
            elem.potential=v
            #periodicrect[hoehe-1,y].fermi_energy=v #not needed!?
        #print "env"
        qcsolver.refresh_environment_contrib()
        #print "qcsolve"
        qcsolution=qcsolver.solve()
        #print "inhom"
        inhom=container.createinhomogeneity()
        #print "solve"
        sol=solver(inhom)
        charges.append(container.charge(sol,charge_operator,charge_elements))
    return numpy.array(charges)

def QuantumCapacityOfGraphene2DModelWithSidegates(hoehe,breite,graphenepos,graphenebreite,vstart,vend,dv,graphenesidegatebreite=None,vsidegate=[(0,0)],graphenesidegate_behavior='metal'):
    #breite=400
    #hoehe=600
    temperature=300
    #graphenepos=hoehe/2-1
    my_gridsize=1e-9
    print "prepare system"
    periodicrect,lapl,grapheneelements,graphenesidegateelementsleft,graphenesidegateelementsright,backgateelements=PeriodicGraphenePatchWithGrapheneSidegatesSiO2Rectangle(breite,hoehe,temperature,graphenepos,graphenebreite,graphenesidegatebreite)
    percont=electrostatics.PeriodicContainer(periodicrect,'y')
    print "solve system"
    solver,inhomogeneity=percont.lu_solver()
    print "init qc"
    #elem1d=[periodicrect[graphenepos,y] for y in range(breite)]#+[periodicrect[0,0]]
    if graphenesidegate_behavior=='metal':
        qcsolver=quantumcapacitance.QuantumCapacitanceSolver(percont,solver,grapheneelements,lapl)
    if graphenesidegate_behavior=='graphene':
        qcsolver=quantumcapacitance.QuantumCapacitanceSolver(percont,solver,grapheneelements+graphenesidegateelementsright+graphenesidegateelementsleft,lapl)

    voltages=numpy.arange(vstart,vend,dv)#-scpotentials   
    print "basisvecs"
    qcsolver.refresh_basisvecs() 
    capacitance_list=[]
    
    for vleftsidegate,vrightsidegate in vsidegate:
        for elem in graphenesidegateelementsleft:
            elem.potential=vleftsidegate
            elem.fermi_energy=vleftsidegate
        
        for elem in graphenesidegateelementsright:
            elem.potential=vrightsidegate       
            elem.fermi_energy=vrightsidegate
    
        print "loop " + str(vrightsidegate)
        charges=QuantumCapacitanceVoltageSweep(qcsolver,percont,lapl,solver,voltages,backgateelements,grapheneelements)
        print "end"
        
        totalcharge=numpy.array([sum(x) for x in charges])
        capacitance2=(totalcharge[2:]-totalcharge[:-2])/len(grapheneelements)*my_gridsize/(2*dv) #1st derivative, 2nd order
        capacitance_list.append(capacitance2)
        
    return voltages[1:-1],capacitance_list

def LoopQuantumCapacitanceWithSidegatesFixedSystem(sidegatevoltages,graphenesidegate_behavior='metal'):
    capacitancequantumlist=[]
    voltages=None
    parameters=[]
    for graphenebreite in (80,):
        print "===Breite " + str(graphenebreite) + "======"
        hoehe=600
        breite=400
        #graphenebreite=10
        grapheneposreal=300
        graphenepos=-grapheneposreal-1
        vstart=-60
        vend=60
        dv=0.5
        graphenesgabstand=30
        sidegatebreite=(breite-2*graphenesgabstand-graphenebreite)/2
        vsidegate=[(sidegatevoltage,sidegatevoltage*sidegatesign) for sidegatevoltage in sidegatevoltages for sidegatesign in (1,-1)]
        #print sidegatebreite
        voltages,capacitancequantum=QuantumCapacityOfGraphene2DModelWithSidegates(hoehe,breite,hoehe+graphenepos,graphenebreite,vstart,vend,dv,sidegatebreite,vsidegate,graphenesidegate_behavior)
        capacitancequantumlist.append(capacitancequantum)
        parameters.append((graphenebreite,vsidegate))
    return voltages,parameters,capacitancequantumlist

def CalcSidegateSaveToFile(sidegatevoltage,graphenesidegate_behavior='metal'):
    voltages,parameters,capacitancequantumlist=LoopQuantumCapacitanceWithSidegatesFixedSystem((sidegatevoltage,),graphenesidegate_behavior)
    numpy.savetxt("qc"+str(sidegatevoltage)+".txt",capacitancequantumlist[0])
