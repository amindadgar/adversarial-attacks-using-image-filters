import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from tensorflow import keras
from inner_genetic_algorithm import inner_algorithm

from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation



class MyProblem(ElementwiseProblem):

    def __init__(self, tf_model):
        n_variables = 5 - 1
        ## since our chromosomes will be permuted, the filter count should be the same as n_variables
        filter_count = n_variables
        super().__init__(
            n_var= n_variables, 
            n_obj=2, 
            n_ieq_constr=0, 
            xl= np.zeros(n_variables), 
            xu=np.full(n_variables, filter_count - 1)
        )
        self.tf_model = tf_model
          

    def _evaluate(self, x, out, *args, **kwargs):
        algorithm = inner_algorithm(self.tf_model, x)
        X_res, F_res = algorithm.start()

        with open('results/filters_with_param_X.txt', 'a+') as file:
            np.savetxt(file, X_res)
            file.write('\n')

        with open('results/filters_with_param_F.txt', 'a+') as file:
            np.savetxt(file, F_res)
            file.write('\n')
        
        out["F"] = F_res[0]



model = keras.models.load_model('model/cifar10_model_90%val_accuracy.h5')
problem = MyProblem(model)

algorithm = NSGA2(
    pop_size=10,
    n_offsprings=10,
    sampling=PermutationRandomSampling(),
    crossover=OrderCrossover(),
    mutation=InversionMutation(),
    eliminate_duplicates=True
)

generation_count = 3
termination = get_termination("n_gen", generation_count)


res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=False,
               verbose=True)

with open('results/final_results_X.txt', 'w') as file:
    np.savetxt(file, res.X)
with open('results/final_results_F.txt', 'w') as file:
    np.savetxt(file, res.F)
