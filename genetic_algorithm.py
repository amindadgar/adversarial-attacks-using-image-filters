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



class MyProblem(ElementwiseProblem):

    def __init__(self, tf_model):

        super().__init__(n_var=3, n_obj=2, n_ieq_constr=0, xl= np.array([0, 0, 0]), xu=np.array([1, 1, 1]))
        self.tf_model = tf_model
          

    def _evaluate(self, x, out, *args, **kwargs):
        fitness = fitness_obj(tf_model=self.tf_model)
        f1 = fitness.fitness_dr(x)
        f2 = fitness.fitness_asr(x)

        out["F"] = [f1, f2]



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

termination = get_termination("n_gen", 3)


res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

