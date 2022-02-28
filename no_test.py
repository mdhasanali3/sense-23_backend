from email import header
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from math import ceil
from flask import abort, send_from_directory
import yaml
# Opening JSON file
# app = Flask(__name__)

# CORS(app)
######     json to yaml
# f = open('data.json')
# print(f['names'])
# # returns JSON object as
# # a dictionary
# data = json.load(f)
 

# f.close()
# dataa = open('data.yaml', 'w')
# yaml.dump(data, dataa, allow_unicode=True)


############



##########   json read check
# Opening JSON file
# f_data = open('data.json')
 
# # returns JSON object as
# # a dictionary
# data_f = json.load(f_data)

# #print(data['epoch'],data['batch'])
#  # Closing file
# f_data.close()


# print("enter  epoch number")
# print(data_f['names'])


#############



######ceil
# total_file=9
# test_size=10#int(input())
# test_size=int(ceil(total_file*(test_size/100)))
# print(test_size)



##############

###########   rest api 

# RESULT_DIRECTORY = "C:/UsershasanDownloads/test/downtest/test"

# @app.route('/get-files/<path:path>',methods = ['GET','POST'])
# def get_files(path):

#     """Download a file."""
#     try:
#         return send_from_directory(RESULT_DIRECTORY, path, as_attachment=True)
#     except FileNotFoundError:
#         abort(404)

# @app.route('/prediction',methods=['POST','GET'])
# def pred_api():
    
#     model_json = 'data.json'#runs/train/exp/weights/best_web_model/model.json'
#     #print(model_json)
#     f_data = open('data.json')
    
#     data_f = json.load(f_data)

#     f_data.close()

#     names=data_f['names']
#     #print(names)
#     pred_json={
#         "model_json": model_json,
#         "names": names
#     }


#     return jsonify(pred_json)


# if __name__ == "__main__":
#     app.run(debug=True)



##############



#############    read csv
import csv

with open('C:/Users/hasan/Downloads/results.csv') as file:
    csreader = csv.reader(file)
    next(csreader)
    data=[]
    for row in csreader:
        data.append({'epoch':row[0],
        'precision':row[4],
        'recall':row[5]
        })

with open ('datanow.json','w') as f:
    json.dump(data,f)
    #print(col['epoch'])
#     if co>10:
#         break
#     co+=1
# for i in rows:
#     print(rows['epoch'])
#print(rows)
file.close()




