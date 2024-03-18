from flask import Flask, request,send_file
from flask_cors import CORS
import os
import infer

app = Flask(__name__)
CORS(app)

# Directory where uploaded images will be stored
UPLOAD_FOLDER = 'datasets/uploads'
RESULT_FOLDER = 'datasets/results'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

@app.route('/synthesis-image', methods=['POST'])
def upload():
    if 'person_image' not in request.files or 'cloth_image' not in request.files:
        return 'Missing images', 400
    
    person_image = request.files['person_image']
    cloth_image = request.files['cloth_image']

    if person_image.filename == '' or cloth_image.filename == '':
        return 'No selected file', 400
    
    person_path = os.path.join(UPLOAD_FOLDER, person_image.filename)
    cloth_path = os.path.join(UPLOAD_FOLDER, cloth_image.filename)

    person_image.save(person_path)
    cloth_image.save(cloth_path)

    img_name=person_image.filename
    c_name=cloth_image.filename    
    infer.main(img_name, c_name)  
    file_name = f"{img_name.replace('.jpg', '')}-{c_name.replace('.jpg', '')}.jpg"
    result_image_path = os.path.join(RESULT_FOLDER, file_name) 
    return send_file(result_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
