from flask import Flask, render_template,request,redirect,url_for
import os
import CATtenf

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(),'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
   return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
      file = request.files['image']
      if 'cdimg' in request.files:
         img=request.files['cdimg']
         print(img)
      print(file)
      filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
      file.save(filename)
      res,sc=CATtenf.run(filename) 
      return render_template('index.html',init=True,imgname=file.filename,answer=res,conf=sc)
    else:
       return redirect(url_for(''))

if __name__ == '__main__':
   app.run(debug = True)