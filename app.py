from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import subprocess
import itertools
import shutil
import random
import glob
import yaml


app = Flask(__name__)

CORS(app)

all_contain_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(all_contain_dir, exist_ok=True)

data_save=True
parameter_save=True

@app.route('/save_annotation/<file_id>',methods=['POST'])
def save_api(file_id):
    image_label_data=request.get_data()   
    curr_dir=os.getcwd()
    os.makedirs('dataset')
    os.chdir('dataset')
    
    with open(file_id, "wb") as code:
        code.write(image_label_data)
    

    #dataset yolov5 formating

        
    print("enter zip file name like data.zip")
    zip_data=file_id
    #input()
    shutil.unpack_archive(zip_data, './')
    #remove('zip_data')
    f = open('data.json')
    
    data = json.load(f)
    f.close()
    dataa = open('data.yaml', 'w')
    yaml.dump(data, dataa, allow_unicode=True)

    total_file=int(len(glob.glob('*'))/2)
    print(total_file)
    #os.chdir('/dataset')
    if os.path.isdir('train/images') is False:
        os.makedirs('train/images')
        os.makedirs('train/labels')
        os.makedirs('valid/images')
        os.makedirs('valid/labels')
        os.makedirs('test/images')
        os.makedirs('test/labels')

    print("enter  train size like 70")
    train_size=70
    #int(input())
    train_size=int(total_file*(train_size/100))
    print(train_size)

    print("enter  valid size like 20")
    valid_size=20
    #int(input())
    valid_size=int(total_file*(valid_size/100))
    print(valid_size)

    print("enter  test size like 10")
    test_size=10#int(input())
    test_size=int(total_file*(test_size/100))
    print(test_size)


    for i in random.sample(range(1,76), train_size):
        try : 
            shutil.move("pro ({}).txt".format(i), 'train/labels')   
            shutil.move("pro ({}).jpg".format(i), 'train/images')   
        except : 
            train_size+=1 
            
    for i in random.sample(range(1,76), valid_size):
        try : 
            shutil.move("pro ({}).txt".format(i), 'valid/labels')   
            shutil.move("pro ({}).jpg".format(i), 'valid/images')   
        except : 
            valid_size+=1  
        
            
    for i in random.sample(range(1,76), test_size):
        try : 
            shutil.move("pro ({}).txt".format(i), 'test/labels')   
            shutil.move("pro ({}).jpg".format(i), 'test/images')   
        except : 
            test_size+=1   

    os.chdir('../')






    return jsonify("done from flask save_annotation")





@app.route('/hyperparameter',methods=['POST'])
def hyp_api():
    
    parameter_json = request.get_json()
   
    with open("parameter_json.json", "w") as outfile:
        json.dump(parameter_json, outfile)

    
    subprocess.run("ls")
    subprocess.run(['python', 'allcontain.py'])

    return jsonify("done from flask parameter")


@app.route('/prediction',methods=['POST','GET'])
def pred_api():
    
    model_json = 'runs/train/exp/weights/best_web_model/model.json'
    print(model_json)
    f_data = open('dataset/data.json')
    
    data_f = json.load(f_data)

    f_data.close()

    names=data_f['names']
    print(names)
    pred_json={
        "model_json": model_json,
        "names": names
    }


    return jsonify(pred_json)


#class names

if __name__ == "__main__":
    app.run(debug=True)