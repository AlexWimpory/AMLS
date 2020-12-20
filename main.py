from A1_feature_pre_processing import save_pre_processed_data as a1_pre_process
from A2_feature_pre_processing import save_pre_processed_data as a2_pre_process
from B1_feature_pre_processing import save_pre_processed_data as b1_pre_process
from B2_feature_pre_processing import save_pre_processed_data as b2_pre_process
from A1_model_predictor import predict as a1_predict
from A2_model_predictor import predict as a2_predict
from B1_model_predictor import predict as b1_predict
from A1_model_trainer import trainer as a1_trainer
from A2_model_trainer import trainer as a2_trainer
from B1_model_trainer import trainer as b1_trainer
from B2_model_trainer import trainer as b2_trainer
from A1_model_evaluator import evaluate as a1_evaluate
from A2_model_evaluator import evaluate as a2_evaluate
from B1_model_evaluator import evaluate as b1_evaluate
from B2_model_evaluator import evaluate as b2_evaluate
import os


def main():
    task = input('Enter a task (A1, A2, B1, B2):\n')
    mode = input('Enter an operation (Train, Evaluate, Predict):\n')

    # Could be re-written using reflection to call the functions
    if task == 'A1':
        os.chdir('A1')
        if mode == 'Train':
            a1_pre_process()
            a1_trainer()
        elif mode == 'Evaluate':
            a1_evaluate()
        elif mode == 'Predict':
            image = input('Enter a image file from celeba_test:\n')
            a1_predict(image)
    elif task == 'A2':
        os.chdir('A2')
        if mode == 'Train':
            a2_pre_process()
            a2_trainer()
        elif mode == 'Evaluate':
            a2_evaluate()
        elif mode == 'Predict':
            image = input('Enter a image file from celeba_test:\n')
            a2_predict(image)
    elif task == 'B1':
        os.chdir('B1')
        if mode == 'Train':
            b1_pre_process()
            b1_trainer()
        elif mode == 'Evaluate':
            b1_evaluate()
        elif mode == 'Predict':
            image = input('Enter a image file from cartoon_set_test:\n')
            b1_predict(image)
    elif task == 'B2':
        os.chdir('B2')
        if mode == 'Train':
            b2_pre_process()
            b2_trainer()
        elif mode == 'Evaluate':
            b2_evaluate()
        elif mode == 'Predict':
            pass


if __name__ == '__main__':
    main()
