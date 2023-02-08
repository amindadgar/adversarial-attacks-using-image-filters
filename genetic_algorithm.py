from fitness import fitness as fitness_obj
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from tensorflow import keras
from inner_genetic_algorithm import inner_algorithm



class MyProblem(ElementwiseProblem):

    def __init__(self, tf_model):
        n_variables = 5
        super().__init__(
            n_var= n_variables, 
            n_obj=2, 
            n_ieq_constr=0, 
            xl= np.zeros(n_variables), 
            xu=np.ones(n_variables)
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



model = keras.models.load_model('../cifar10_model_90%val_accuracy.h5')
problem = MyProblem(model)

algorithm = NSGA2(
    pop_size=10,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=0.5 ,eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 5)


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
