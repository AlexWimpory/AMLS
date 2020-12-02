import os

from A1_model_trainer import trainer as a1_trainer
from A2_model_trainer import trainer as a2_trainer
from B1_model_trainer import trainer as b1_trainer
from B2_model_trainer import trainer as b2_trainer
from A1_model_evaluator import evaluate as a1_evaluate
from A2_model_evaluator import evaluate as a2_evaluate
from B1_model_evaluator import evaluate as b1_evaluate
from B2_model_evaluator import evaluate as b2_evaluate


def main():
    task = input('Enter a task (A1, A2, B1, B2)')
    mode = input('Enter an operation (Train, Evaluate, Predict)')

    # Could be re-written using reflection to call the functions
    if task == 'A1':
        os.chdir('A1')
        if mode == 'Train':
            a1_trainer()
        elif mode == 'Evaluate':
            a1_evaluate()
        elif mode == 'Predict':
            pass
    elif task == 'A2':
        os.chdir('A2')
        if mode == 'Train':
            a2_trainer()
        elif mode == 'Evaluate':
            a2_evaluate()
        elif mode == 'Predict':
            pass
    elif task == 'B1':
        os.chdir('B1')
        if mode == 'Train':
            b1_trainer()
        elif mode == 'Evaluate':
            b1_evaluate()
        elif mode == 'Predict':
            pass
    elif task == 'B2':
        os.chdir('B2')
        if mode == 'Train':
            b2_trainer()
        elif mode == 'Evaluate':
            b2_evaluate()
        elif mode == 'Predict':
            pass


if __name__ == '__main__':
    main()
