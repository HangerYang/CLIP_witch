import numpy as np

model_names = ['target_dog_intend_cat_targetn_1_restart_4_attkiter_250_classification_1000000']
def show_results(model_name, result_type='similarity'):
    arr = np.load("save_verify_text_with_csv/{}_{}.npy".format(model_name, result_type))

    for epoch in range(len(arr)):
        print('epoch %d'%(epoch+1))
        print(arr[epoch])

for model in model_names:
    show_results(model)