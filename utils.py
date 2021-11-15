import os
import sys

def set_train_path(roadnet,method,mode,metric,comment):
    """
    Create a new save path with an incremental integer, also considering previously created save paths
    """
    train_path = os.path.join(os.getcwd(),'save',roadnet,method,mode,metric,'')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    dir_content = os.listdir(train_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'
    data_path = os.path.join(train_path,'result_'+new_version+'_'+comment,'')
    print(data_path)
    os.makedirs(os.path.dirname(data_path),exist_ok=True)
    return data_path

def set_test_path(roadnet,method,metric, model_n):
    model_root_path = os.path.join(os.getcwd(),'save',roadnet, method, 'train', metric,'')
    dir_content = os.listdir(model_root_path)
    if dir_content:
        for name in dir_content:
            if int(name.split("_")[1]) == model_n:
                path = name
                break
        
        test_path = os.path.join(os.getcwd(),'save',roadnet, method, 'test', metric, path,'')
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        model_path = os.path.join(model_root_path, path, '')
        return model_path, test_path
    else:
        print("path not exists")
