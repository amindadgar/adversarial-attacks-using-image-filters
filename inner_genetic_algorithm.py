from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from fitness import fitness as fitness_obj
import numpy as np
from filters.filters import img_filters



class inner_algorithm():
    def __init__(self, tf_model, filters_chromosome) -> None:
        self.filters_chromosome = filters_chromosome
        self.tf_model = tf_model


    def start(self):
        """
        Start the inner algorithm

        Returns:
        ---------
        res.X : array_like
            the pareto set of the filters and their parameters
        res.F : array_like
            the fitness values of them
        """
        ## for each filter we have two parameters
        ## so we set the number of paramters multiplied to 2
        problem = self.problem_filters_parameter(self.tf_model, len(self.filters_chromosome) * 2, self.filters_chromosome )
        
        ## initialize parameters with value of 1
        pop_size = 5
        ## the initial parameters of the filters are always equal to 1
        initial_generation = np.ones((pop_size, len(self.filters_chromosome)*2))
        # print(initial_generation.shape, '\n\n')

        algorithm = NSGA2(
            pop_size=pop_size,
            n_offsprings=10,
            sampling=initial_generation,
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=0.5 ,eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 3)

        # print('Running inner algorithm')

        res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=False,
               verbose=False)
        
        # print('Inner algorithm finished!')
        # print(self.filters_chromosome.shape, res.X.shape)
        ## append the parameters to the image filter chromosome
        new_chromosomes = np.append(np.repeat(np.expand_dims(self.filters_chromosome, axis=0), len(res.X), axis=0) ,res.X ,axis=1)
        new_chromosomes_fitness = res.F
        return new_chromosomes, new_chromosomes_fitness
    


    class problem_filters_parameter(ElementwiseProblem):

        def __init__(self, tf_model, num_filters_parameters, chromosome_filter):

            super().__init__(n_var=num_filters_parameters, 
                            n_obj=2, 
                            n_ieq_constr=0, 
                            xl= np.zeros(num_filters_parameters), 
                            xu=np.ones(num_filters_parameters))
                
            self.tf_model = tf_model
            self.chromosome_filter = chromosome_filter
    
        def _get_filter_function(self, filter_gene, filter_alpha, filter_strength):
            """
            get the filter function
            """
            ## to choose the filter
            # if 0 < filter_gene <= 0.25:
            if filter_gene == 0 :
                filter_name = 'kelvin'
            # elif 0.25 < filter_gene <= 0.5:
            elif filter_gene == 1:
                filter_name = 'clarendon'
            # elif 0.5 < filter_gene <= 0.75:
            elif filter_gene == 2:
                filter_name = 'moon'
            # elif 0.75 < filter_gene <= 1:
            elif filter_gene == 3:
                # filter_name = 'sharpening'
                filter_name = 'gingham'
            else:
                raise ValueError(f'filter chromosome value should be integer values between 0 to 3, entered value: {filter_gene}')
            
            filter_function = img_filters(filter_alpha, filter_strength, filter_name )
            return filter_function

        def _extract_filter_functions(self, chromosome_parameters, chromosome_filter):
            """
            Extract multiple filters from the chromosome
            """
            # print(chromosome_parameters)
            # print(chromosome_filter, '\n\n')
            ## the index of parameters of each filter
            parameter_index = {
                0: [0, 1],
                1: [2, 3],
                2: [4, 5],
                3: [6, 7]
            }
            filters_function = []
            ## initialize each filter function
            for i in range(4):
                ## the parametrs
                alpha_index = parameter_index[i][0]
                strength_index = parameter_index[i][1]
                ## the function creation
                filters_function.append(
                    self._get_filter_function(chromosome_filter[i],
                    chromosome_parameters[alpha_index], 
                    chromosome_parameters[strength_index]
                    )
                )
            
            return filters_function    

        def _evaluate(self, x, out, *args, **kwargs):
            filters_function = self._extract_filter_functions(x, self.chromosome_filter)
            fitness = fitness_obj(tf_model=self.tf_model, filters_functions= filters_function)
            f1 = fitness.fitness_dr()
            f2 = fitness.fitness_asr()

            out["F"] = [f1, f2]
