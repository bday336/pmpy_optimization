import numpy as np
from npy_append_array import NpyAppendArray
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
        # self.pathlist = []

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

            self.initpath = self.trial_solution
            self.initeval = self.dyn_functions[0](self.initpath, self.vt, self.mu, self.dt)
            self.inithval = self.dyn_functions[1](self.initpath, self.vt, self.dt)

            # Generate Data Files
            np.save("pmpy_gdopt_100_pdat.npy", np.array([self.initpath]))
            np.save("pmpy_gdopt_100_edat.npy", np.array([self.initeval]))
            np.save("pmpy_gdopt_100_hdat.npy", np.array([self.inithval]))

            self.egrad = self.dyn_funcgrads[0](self.initpath, self.vt, self.mu, self.dt)
            self.hgrad = self.dyn_funcgrads[1](self.initpath, self.vt, self.dt)
            self.modgrad = self.egrad - (np.dot(self.egrad.flatten(),self.hgrad.flatten()))/(np.dot(self.hgrad.flatten(),self.hgrad.flatten()))*self.hgrad

            # For closed-loop path
            self.pathpart = list(self.initpath[0:self.nump-1] - self.learning_rate*self.modgrad)
            self.pathpart.append(self.pathpart[0])

            # Perform step 1
            self.finalpath = self.pathpart
            self.finaleval = self.dyn_functions[0](self.finalpath, self.vt, self.mu, self.dt)
            self.finalhval = self.dyn_functions[1](self.finalpath, self.vt, self.dt)
            print(self.finaleval - self.initeval)

            # Optimization
            for a in range(1,self.maximum_iteration):
                if abs(self.finaleval - self.initeval) <= self.tolerance or self.finaleval > self.initeval:
                    break
                else:
                    if a % 1000 == 0:
                        print(a)
                        NpyAppendArray("pmpy_gdopt_100_pdat.npy").append(np.array([self.finalpath]))
                        NpyAppendArray("pmpy_gdopt_100_edat.npy").append(np.array([self.finaleval]))
                        NpyAppendArray("pmpy_gdopt_100_hdat.npy").append(np.array([self.finalhval]))
                        print(self.finaleval - self.initeval)

                    self.initpath = self.finalpath
                    self.initeval = self.finaleval
                    self.inithval = self.finalhval

                    self.egrad = self.dyn_funcgrads[0](self.initpath, self.vt, self.mu, self.dt)
                    self.hgrad = self.dyn_funcgrads[1](self.initpath, self.vt, self.dt)
                    self.modgrad = self.egrad - (np.dot(self.egrad.flatten(),self.hgrad.flatten()))/(np.dot(self.hgrad.flatten(),self.hgrad.flatten()))*self.hgrad

                    # For closed-loop path
                    self.pathpart = list(self.initpath[0:self.nump-1] - self.learning_rate*self.modgrad)
                    self.pathpart.append(self.pathpart[0])

                    # Perform step 1
                    self.finalpath = self.pathpart
                    self.finaleval = self.dyn_functions[0](self.finalpath, self.vt, self.mu, self.dt)
                    self.finalhval = self.dyn_functions[1](self.finalpath, self.vt, self.dt)

            print("Simulation run completed!")
            self._have_run = True
        else:
            raise NotImplementedError("Clear data with clear_data() before rerunning optimization")

    