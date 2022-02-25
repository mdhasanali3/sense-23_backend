from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from math import ceil
from flask import abort, send_from_directory
import yaml
# Opening JSON file
app = Flask(__name__)

CORS(app)
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



RESULT_DIRECTORY = "C:/UsershasanDownloads/test/downtest/test"

@app.route('/get-files/<path:path>',methods = ['GET','POST'])
def get_files(path):

    """Download a file."""
    try:
        return send_from_directory(RESULT_DIRECTORY, path, as_attachment=True)
    except FileNotFoundError:
        abort(404)

@app.route('/prediction',methods=['POST','GET'])
def pred_api():
    
    model_json = 'data.json'#runs/train/exp/weights/best_web_model/model.json'
    #print(model_json)
    f_data = open('data.json')
    
    data_f = json.load(f_data)

    f_data.close()

    names=data_f['names']
    #print(names)
    pred_json={
        "model_json": model_json,
        "names": names
    }


    return jsonify(pred_json)


if __name__ == "__main__":
    app.run(debug=True)