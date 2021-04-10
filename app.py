import os, glob
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from srfbn.test import main
from srrescgan.predict import main_srrescgan
import numpy as np

UPLOAD_FOLDER = 'static/uploads/'
DOWNLOAD_FOLDER = 'static/downloads/'
ALLOWED_EXTENSIONS = {'jpg', 'png', '.jpeg'}
app = Flask(__name__, static_url_path="/static")

# APP CONFIGURATIONS
app.config['SECRET_KEY'] = 'opencv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 6mb
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        model = request.form.get('comp_select')
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            ul = glob.glob('./static/uploads/*')
            for f in ul:
                os.remove(f)
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            process_file(os.path.join(UPLOAD_FOLDER, filename), filename, model)
            data = {
                "processed_img": 'static/downloads/' + model+'_'+filename,
                "uploaded_img": 'static/uploads/' + filename
            }
            return render_template("index.html", data=data)
    return render_template('index.html')


def process_file(path, filename, model):
    if model == 'SRFBN':
        main()
    if model == 'SRResCGAN':
        main_srrescgan()


if __name__ == '__main__':
    app.run()
