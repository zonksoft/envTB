#TODO: BulkGrapheneWithTemperature.Ef_interp auf genauigkeit testen! min/maxEf parametrisieren

import sympy.mpmath
import scipy.optimize
from .common import Constants
import numpy
import math
import scipy.interpolate

class QuantumCapacitanceSelfConsistency:
    container=0
    charge_operator=0
    solver=0
    convergence_tol=0
    
    def __init__(self,container,charge_operator,convergence_tol=1e-2,solver=None):
        if solver!=None:
            self.solver=solver
        else:
            self.solver,inhomogeneity=container.lu_solver()
        self.charge_operator=charge_operator
        self.container=container
        self.convergence_tol=convergence_tol
        
        
    def self_consistency_cycle(self,elements):
        
        rectangle_elementnumbers_range=self.container.rectangle_elementnumbers_range()
        
        
        
        while True:
            inhom=self.container.createinhomogeneity()
            solution=self.solver(inhom)
            
            diff=0
            
            for elem in elements:
                
                charge=self.container.charge(solution,self.charge_operator,[elem])
                old=elem.potential
                elem.potential=elem.fermi_energy-elem.fermi_energy_charge_dependence(charge)/Constants.elem_charge
                diff+=math.fabs(elem.potential-old)
            print diff
            if diff < self.convergence_tol:
                break
            
        inhom=self.container.createinhomogeneity()
        solution=self.solver(inhom)
        
        return solution
        #Solution can also be obtained by looking at the potential value of the elements
        
class QuantumCapacitanceSolver:
    container=None
    solver=None
    elements=None
    charge_operator=None
    
    withnopot=None
    m=None
    
    def __init__(self,container,solver,elements,charge_operator):
        self.container=container
        self.solver=solver
        self.elements=elements
        self.charge_operator=charge_operator
        
    def refresh_environment_contrib(self):
        self.reset_potential()
        inhom=self.container.createinhomogeneity()
        self.withnopot=self.container.charge(self.solver(inhom),self.charge_operator,self.elements)
        
    def refresh_basisvecs(self,status=False):
        self.reset_potential()
            
        if self.withnopot==None:
            self.refresh_environment_contrib()
            
        basisvecs=[]
        for elem in self.elements:
            if status:
                print elem.index()
            elem.potential=1
            #periodicrect[0,0].potential=1#####################
            inhom=self.container.createinhomogeneity()
            solution=self.solver(inhom)
            basisvecs.append(self.container.charge(solution,self.charge_operator,self.elements)-self.withnopot)
            elem.potential=0
            #periodicrect[0,0].potential=0###################
            
        self.m=numpy.array(basisvecs)
        
    def f(self,potvalues):
        for x,elem in zip(potvalues,self.elements):
            elem.potential=x
        #periodicrect[0,0].potential=x############################    
        return self.functiontominimize()
    
    def functiontominimize(self): #actually the root, not the minimum
        potvec=numpy.array([elem.potential for elem in self.elements])
        fermivec=numpy.array([elem.fermi_energy for elem in self.elements])
        sqrtvec=[elem.fermi_energy_charge_dependence(charge)/Constants.elem_charge for elem,charge in zip(self.elements,numpy.dot(potvec,self.m)+self.withnopot)]
        return -fermivec+potvec+sqrtvec
    
    def solve(self):
        potvec=numpy.array([elem.potential for elem in self.elements])
        loesung=scipy.optimize.root(self.f,potvec,method='lm')
        return loesung
    
    def get_charge(self):
        inhom=self.container.createinhomogeneity()
        solution=self.solver(inhom)
        charge=self.container.charge(solution,self.charge_operator,self.elements)
        return charge
        
    def get_potential(self):
        potvec=numpy.array([elem.potential for elem in self.elements])
        return potvec
    
    def reset_potential(self):
        for elem in self.elements:
            elem.potential=0

class FermiEnergyChargeDependence:
    @staticmethod
    def bulk_graphene(charge,grid_height=1e-9): #grid_height*volume density = area density
        return -Constants.v_fermi*Constants.hbar*math.copysign(math.sqrt(math.pi*math.fabs(charge*grid_height)/Constants.elem_charge),charge)
    
    @staticmethod
    def no_dependence(charge):
        return 0.
    
class BulkGrapheneWithTemperature:
    T=None
    grid_height=None
    interp=None
    minEf=-1.5e-18
    maxEf=1.5e-18
    Eftol=1e-25 #for root search
    dEf=1e-22 #for interpolation
    
    def __init__(self,T,grid_height):
        self.T=T
        self.grid_height=grid_height
       
        Ef_grid=numpy.arange(self.maxEf,self.minEf,-self.dEf)
        Q_grid=[self.Q(i,T)/grid_height for i in Ef_grid]
        self.interp=scipy.interpolate.interp1d(Q_grid,Ef_grid)
               
    def J1(self,eta):
        return -float(sympy.mpmath.polylog(2,-sympy.exp(eta)))
    
    def n(self,Ef,T):
        return 2/math.pi*(Constants.k_B*T/(Constants.hbar*Constants.v_fermi))**2*self.J1(Ef/(Constants.k_B*T))
    
    def p(self,Ef,T):
        return 2/math.pi*(Constants.k_B*T/(Constants.hbar*Constants.v_fermi))**2*self.J1(-Ef/(Constants.k_B*T))
    
    def Q(self,Ef,T):
        return Constants.elem_charge*(self.p(Ef,T)-self.n(Ef,T))
    
    def Ef(self,charge,T=None):
        if T==None:
            T=self.T
        return scipy.optimize.brentq(lambda myEf: self.Q(myEf,T)-charge*self.grid_height,self.minEf,self.maxEf,xtol=self.Eftol)#grid_height*volume density = area density
    
    def Ef_interp(self,charge):
        return self.interp(charge)