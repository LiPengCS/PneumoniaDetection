import os
from uuid import uuid4
from flask import Flask, render_template, request

import visualize
import utils
from PIL import Image

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Trial.html')


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        
        name = upload.filename
        destination = "/".join([target, name])
        print ("Accept incoming file:", name)
        print ("Save it to:", destination)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    img = utils.load_image("./images/" + name)
    y_pred, prob, visualize_result = visualize.visualize(img)
    file_names = {}
    file_names["kernel"] = []
    pred = []
    for name, result in visualize_result.items():
        if name != "kernel":
            Image.fromarray(result).save("./static/result/{}.png".format(name))
            file_names[name] =  "{}.png".format(name)
        else:
            for i, res in enumerate(result):
                Image.fromarray(res).save("./static/result/{}_{}.png".format(name, i))
                file_names["kernel"].append("{}_{}.png".format(name, i))
    if y_pred == 0:
        pred.append("The patient has no pneumonia :)")
    else:
        pred.append("The patient has a high risk of getting pneumonia!")
    temp = str(round(prob[0]*100, 2)) + "%"
    pred.append(temp)
    return render_template("results.html", results=file_names,predict=pred)

if __name__ == '__main__':
	app.run(debug=True)