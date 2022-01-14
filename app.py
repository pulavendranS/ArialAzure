from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from pathlib import Path
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2 as cv

app = Flask(__name__)

dic = {0 : 'Field', 1 : 'Forest', 2 :"Grass", 3:'Industry', 4:'Parking', 5:'Resident', 6:'RiverLake'}


f = Path("model/gab_model_structure.json")
model_structure = f.read_text()
model = model_from_json(model_structure)
model.load_weights("model/gab_model_weights.h5")

def predict_label(img_path):
	img = cv.imread(img_path,0) #reading image
	gabor_1 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F)
	filtered_img_1 = cv.filter2D(img, cv.CV_8UC3, gabor_1)
	gabor_2 = cv.getGaborKernel((18, 18), 1.5, np.pi/4, 5.0, 1.5, 0, ktype=cv.CV_32F)
	filtered_img_2 = cv.filter2D(filtered_img_1, cv.CV_8UC3, gabor_2)
	cv.imwrite("gabor_imgs/test.jpg",filtered_img_2)
	img_1 = image.load_img("gabor_imgs/test.jpg", target_size=(256, 256))
	new_img = image.img_to_array(img_1)
	img_exp = np.expand_dims(new_img, axis=0)
	p = model.predict(img_exp)
	x = np.argmax(p)
	return dic[x]


# routes
@app.route("/", methods=['GET', 'POST'])
def kuch_bhi():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return "About You..!!!"


@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)	
		p = predict_label(img_path)

	return render_template("home.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)