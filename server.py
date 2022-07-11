import api_face
import face_recognize_api
import os
import numpy
import cv2
from PIL import Image
from flask import Flask, request, Response, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import ssl

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    # Put any other methods you need here
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/', methods=['POST'])
def upload_file():

    nome = request.form['nome']
    print(nome)

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Nenhuma imagem selecionada')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = nome+'.jpg'
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], nome+'.jpg'))        
        # print('upload_image filename: ' + filename)
        flash('Imagem enviada com sucesso')
        print('filename', filename)
        print(app.config['UPLOAD_FOLDER'])
        return render_template('index.html', filename=filename)
    else:
        flash('Imagens válidas: - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>', methods=['GET'])
def display_image(filename):
    return redirect(url_for('static', filename='images/' + filename), code=301)

@app.route('/train')
def train():
    try:
        print('Started Training')
        face_recognize_api.train('./data/face-images')
        print('finished Training')
        return Response('Training finished.')
    except Exception as e:
        print('POST /image error: %e' % e)
        return e


@app.route('/reconhecer')
def local():
    return render_template('local.html')
    #return Response(open('./static/templates/local.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


# @app.route('/test')
# def test():
#     PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
#     TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

#     image = Image.open(TEST_IMAGE_PATHS[0])
#     objects = object_detection_api.get_objects(image)

#     return objects


@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image

        print("image_file", image_file)

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)

        uploadWidth = request.form.get('uploadWidth')
        if uploadWidth is None:
            uploadWidth = 800.0
        else:
            uploadWidth = float(uploadWidth)

        uploadHeight = request.form.get('uploadHeight')
        if uploadHeight is None:
            uploadHeight = 600.0
        else:
            uploadHeight = float(uploadHeight)
        # finally run the image through tensor flow object detection`
        # image_object = cv2.imread('face-images/mike/tuan.jpg')
        image_object = cv2.imdecode(numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
        response = api_face.predict(image_object, threshold, uploadWidth, uploadHeight)

        print('response', response)
        return response

    except Exception as e:
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
    # without SSL
   # app.run(debug=True, host='0.0.0.0', threaded=True)

    #  app.run(debug=True, host='0.0.0.0', ssl_context=('example.com+5.pem', 'example.com+5-key.key'))
    # app.run(debug=True, host='0.0.0.0', threaded=True, ssl_context=('ssl/cert.crt', 'ssl/key.key'))

    # app.run(debug=True, host='0.0.0.0', ssl_context='adhoc')
    app.run(ssl_context=("cert.pem", "key.pem"))
