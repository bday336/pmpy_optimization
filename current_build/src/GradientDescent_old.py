import numpy as np
from src.integrator_files.integrator_bank import gausss1, gausss2, gausss3, rads2, rads3


# Simulation class setup

class GradientDescent:
    """
    A class used to perform gradient descent

    ...

    Attributes
    ---------- 
    learning_rate : float
        the step size of the gradient descent

    tolerance : float
        the convergence tolerance to terminate process
    
    maximum_iteration : int
        the maximum number of iterations to run

    

    Methods
    -------
    set_initial_conditions(system_ics)
        Inputs user given initial conditions of system (trial solution)

    clear_data()
        Clears any simulation data stored in simulation object

    run()
        Runs optimization once given all necessary information

    output_data()
        Outputs optimization data to file with name:
            pmpy_gdopt_tmax{self.tmax}_dt{self.dt}.npy
    """

    def __init__(self, dyn_functions, dyn_funcgrads, trial_solution, system_params, learning_rate, tolerance, maximum_iteration):
        """
        Parameters
        ----------
        dyn_functions : list of functions
            the functions describing system (metric and holonomy functions)

        dyn_funcgrads : list of functions
            the discrete gradients of the functions describing system 

        trial_solution : array
            the discrete trial solution curve in control space

        learning_rate : float
            the step size of the gradient descent

        tolerance : float
            the convergence tolerance to terminate process
        
        maximum_iteration : int
            the maximum number of iterations to run

        """

        # Functions specifying system
        self.dyn_functions = dyn_functions

        # Gradients specifying system
        self.dyn_funcgrads = dyn_funcgrads

        # Trial solution
        self.trial_solution = trial_solution

        # Data Containers
        self.elist = []
        self.hlist = []
        self.pathlist = []

        # System Parameters
        self.system_params  = system_params

        # Learning Rate
        self.learning_rate = learning_rate

        # Terminate conditions
        self.tolerance = tolerance
        self.maximum_iteration = maximum_iteration

        # Internal Flags
        self._have_run = False
        self._have_ics = True

    def clear_data(self):
        """
        Clears any simulation data stored in simulation object

        """
        if self._have_ics:
            self.elist = []
            self.hlist = []
            self.pathlist = []
            self._have_run = False
        else:
            raise NotImplementedError("Must provide initial conditions via set_initial_conditions(), no data to clear")
        

    def run(self):
        """
        Runs simulation once given all necessary information

        Raises
        ----------
        NotImplementedError
            If no initial conditions have been provided
        """

        if self._have_ics and not self._have_run:

            # Initialize first step (Constant_H)
            self.vt , self.mu , self.dt , self.nump = self.system_params

            self.pathlist.append(self.trial_solution)
            self.elist.append(self.dyn_functions[0](self.pathlist[-1], self.vt, self.mu, self.dt))
            self.hlist.append(self.dyn_functions[1](self.pathlist[-1], self.vt, self.dt))

            self.egrad = self.dyn_funcgrads[0](self.pathlist[-1], self.vt, self.mu, self.dt)
            self.hgrad = self.dyn_funcgrads[1](self.pathlist[-1], self.vt, self.dt)
            self.modgrad = self.egrad - (np.dot(self.egrad.flatten(),self.hgrad.flatten()))/(np.dot(self.hgrad.flatten(),self.hgrad.flatten()))*self.hgrad

            # For loop
            self.pathpart = list(self.pathlist[-1][0:self.nump-1] - self.learning_rate*self.modgrad)
            self.pathpart.append(self.pathpart[0])

            # Perform step 1
            self.pathlist.append(self.pathpart)
            self.elist.append(self.dyn_functions[0](self.pathlist[-1], self.vt, self.mu, self.dt))
            self.hlist.append(self.dyn_functions[1](self.pathlist[-1], self.vt, self.dt))

            # Optimization
            for a in range(self.maximum_iteration):
                if abs(self.elist[-1] - self.elist[-2]) <= self.tolerance or self.elist[-1] > self.elist[-2]:
                    break
                else:
                    if a % 100 == 0:
                        print(a)
                        print(self.elist[-1] - self.elist[-2])
                    self.egrad = self.dyn_funcgrads[0](self.pathlist[-1], self.vt, self.mu, self.dt)
                    self.hgrad = self.dyn_funcgrads[1](self.pathlist[-1], self.vt, self.dt)
                    self.modgrad = self.egrad - (np.dot(self.egrad.flatten(),self.hgrad.flatten()))/(np.dot(self.hgrad.flatten(),self.hgrad.flatten()))*self.hgrad

                    # For loop
                    self.pathpart = list(self.pathlist[-1][0:self.nump-1] - self.learning_rate*self.modgrad)
                    self.pathpart.append(self.pathpart[0])

                    # Perform step 1
                    self.pathlist.append(self.pathpart)
                    self.elist.append(self.dyn_functions[0](self.pathlist[-1], self.vt, self.mu, self.dt))
                    self.hlist.append(self.dyn_functions[1](self.pathlist[-1], self.vt, self.dt))

            print("Simulation run completed!")
            self._have_run = True
        else:
            raise NotImplementedError("Clear data with clear_data() before rerunning optimization")

    def output_data(self):
        """
        Outputs simulation data to file with name:
            pmpy_gdopt_{data}_dt{self.dt}.npy

        Raises
        ----------
        NotImplementedError
            If simulation has not been run, i.e. no data generated
        """

        if self._have_run:
            np.save("pmpy_gdopt_pathdat_dt{}".format(str(self.dt)), self.pathlist)
            np.save("pmpy_gdopt_edat_dt{}".format(str(self.dt)), self.elist)
            np.save("pmpy_gdopt_hdat_dt{}".format(str(self.dt)), self.hlist)
        else:
            raise NotImplementedError("Must use run() to generate data")

    