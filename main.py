from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from camera2 import VideoCamera2
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import cv2
import numpy as np
import threading
import os
import time
import shutil
import imagehash
import PIL.Image
from PIL import Image
from PIL import ImageTk
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="atm_face"
)

app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff11=open("img.txt","w")
    ff11.write("1")
    ff11.close()

    
        

    return render_template('index.html',msg=msg,act=act)

@app.route('/verify_card',methods=['POST','GET'])
def verify_card():
    cnt=0
    act=""
    msg=""

    ff11=open("facest.txt","w")
    ff11.write("")
    ff11.close()
    
    if request.method=='POST':
        card=request.form['card']
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where card=%s",(card, ))
        cnt = mycursor.fetchone()[0]
        if cnt>0:
            msg="success"
            session['username'] = card
            ff2=open("un.txt","w")
            ff2.write(card)
            ff2.close()
            return redirect(url_for('verify_face'))
       
            
        else:
            msg="Card No. is wrong!"
            print("Incorrect")
        

    return render_template('verify_card.html',msg=msg,act=act)

#########################

@app.route('/register',methods=['POST','GET'])
def register():
    result=""
    act=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        bank=request.form['bank']
        branch=request.form['branch']
        card=request.form['card']
        account=request.form['accno']
        uname=request.form['username']
        password=request.form['password']

        aadhar1=request.form['aadhar1']
        aadhar2=request.form['aadhar2']
        aadhar3=request.form['aadhar3']

        face_st=request.form['face_st']
        
        
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM register where card=%s",(card, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO register(id, name, mobile, email, address,  bank, accno, branch, card, deposit, username, password, rdate, aadhar1, aadhar2, aadhar3, face_st, fimg) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, address, bank, account, branch, card, '10000', uname, password, rdate, aadhar1, aadhar2, aadhar3, face_st, '')
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            if face_st=="1":
                return redirect(url_for('add_photo',vid=maxid))
            #if mycursor.rowcount==1:
            #    result="Registered Success"
            else:
                return redirect(url_for('index',act='success'))
        else:
            result="Card No. already Exist!"
    return render_template('register.html',result=result)

@app.route('/login_admin', methods=['POST','GET'])
def login_admin():
    result=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            result="Incorrect USername or Password!!!"
                
    
    return render_template('login_admin.html',result=result)

@app.route('/admin',methods=['POST','GET'])
def admin():
    msg=""
    email=""
    mess=""
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    mycursor = mydb.cursor()
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        address=request.form['address']
        branch=request.form['branch']
        aadhar=request.form['aadhar']

        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        
        
        mycursor.execute("SELECT count(*) FROM register where aadhar1=%s",(aadhar, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            str1=str(maxid)
            ac=str1.rjust(4, "0")
            account="223344"+ac

            xn=randint(1000, 9999)
            rv1=str(xn)
            xn2=randint(1000, 9999)
            rv2=str(xn2)
            card=rv1+ac+rv2
            bank="SBI"

            xn3=randint(1000, 9999)
            pinno=str(xn3)
            
            
            sql = "INSERT INTO register(id, name, mobile, email, address,  bank, accno, branch, card, deposit,password, rdate, aadhar1) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            val = (maxid, name, mobile, email, address, bank, account, branch, card, '10000',pinno, rdate, aadhar)
            vid=str(maxid)
            mycursor.execute(sql, val)
            mydb.commit()
            mess="Dear "+name+", Your Bank Account created, Account No.:"+account+", Card No."+card
            #url="http://iotcloud.co.in/testmail/sendmail.php?email="+email+"&message="+message
            #webbrowser.open_new(url)
            msg="success"
            #return redirect(url_for('add_photo',vid=maxid)) 
        else:
            msg="Already Exist!"

    mycursor.execute("SELECT amount FROM admin WHERE username='admin'")
    value = mycursor.fetchone()[0]
    
    return render_template('admin.html',msg=msg,value=value,email=email,mess=mess,vid=vid)

def getImagesAndLabels(path):

    
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids


@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    cursor = mydb.cursor()
    
    
    vid = request.args.get('vid')
    
    cursor.execute("SELECT * FROM register where id=%s",(vid,))
    value = cursor.fetchone()
    name=value[1]
    
    ff=open("user.txt","w")
    ff.write(name)
    ff.close()

    ff=open("user1.txt","w")
    ff.write(vid)
    ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from vt_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1="User."+vid+"."+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface="User."+vid+"."+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update register set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        ##########
        
        ##Training face
        # Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # function to get the images and label data
        

        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))






        #################################################
        return redirect(url_for('view_photo',vid=vid,act='success'))
        
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM register")
    data = cursor.fetchall()
    return render_template('add_photo.html',data=data, vid=vid)

@app.route('/view_cus',methods=['POST','GET'])
def view_cus():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    value = mycursor.fetchall()
    return render_template('view_cus.html', result=value)

###Preprocessing
@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            ######
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
            #shutil.copy('static/images/11.png', 'static/process4/'+rs[2])
       
        return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', result=value,vid=vid)


#FRCNN
def FRCNN(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def model():       
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = int(options.num_rois)

    if C.network == 'resnet50':
            num_features = 1024
    elif C.network == 'vgg':
            num_features = 512

    if K.common.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
    else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)


    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print(f'Loading weights from {C.model_path}')
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    all_imgs = []

    classes = {}

    bbox_threshold = 0.8

    visualise = True

    for idx, img_name in enumerate(sorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                    continue
            print(img_name)
            st = time.time()
            filepath = os.path.join(img_path,img_name)

            img = cv2.imread(filepath)

            X, ratio = format_img(img, C)

            if K.common.image_dim_ordering() == 'tf':
                    X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)
            

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                    ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                    if ROIs.shape[1] == 0:
                            break

                    if jk == R.shape[0]//C.num_rois:
                            #pad R
                            curr_shape = ROIs.shape
                            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                            ROIs_padded[:, :curr_shape[1], :] = ROIs
                            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                            ROIs = ROIs_padded

                    [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                    for ii in range(P_cls.shape[1]):

                            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                                    continue

                            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                            if cls_name not in bboxes:
                                    bboxes[cls_name] = []
                                    probs[cls_name] = []

                            (x, y, w, h) = ROIs[0, ii, :]

                            cls_num = np.argmax(P_cls[0, ii, :])
                            try:
                                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                                    tx /= C.classifier_regr_std[0]
                                    ty /= C.classifier_regr_std[1]
                                    tw /= C.classifier_regr_std[2]
                                    th /= C.classifier_regr_std[3]
                                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                            except:
                                    pass
                            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                            probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                    bbox = np.array(bboxes[key])

                    new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                    for jk in range(new_boxes.shape[0]):
                            (x1, y1, x2, y2) = new_boxes[jk,:]

                            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                            textLabel = f'{key}: {int(100*new_probs[jk])}'
                            all_dets.append((key,100*new_probs[jk]))

                            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                            textOrg = (real_x1, real_y1-0)

                            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)


##Classification
#CNN
def CNN():
    channel, height, width = image_shape
    input_layer = lasagne.layers.InputLayer(shape=(None, channel, height, width), input_var=variable)
    conv1 = lasagne.layers.Conv2DLayer(incoming=input_layer, num_filters=8, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    conv2 = lasagne.layers.Conv2DLayer(incoming=conv1, num_filters=8, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    pool1 = lasagne.layers.Pool2DLayer(incoming=conv2, pool_size=(2, 2), stride=(2, 2), pad=0)
    drop1 = lasagne.layers.DropoutLayer(incoming=pool1, p=droput)
    conv3 = lasagne.layers.Conv2DLayer(incoming=drop1, num_filters=16, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    conv4 = lasagne.layers.Conv2DLayer(incoming=conv3, num_filters=16, filter_size=(3, 3), pad=0, stride=(1, 1), nonlinearity=activation, W=weight, b=bias)
    pool2 = lasagne.layers.Pool2DLayer(incoming=conv4, pool_size=(2, 2), stride=(2, 2), pad=0)
    drop2 = lasagne.layers.DropoutLayer(incoming=pool2, p=droput)
    fc = lasagne.layers.DenseLayer(incoming=drop2, num_units=len(LABELS), nonlinearity=classifier, W=weight, b=bias)
    return fc

def load_network_from_model(network, model):
    with open(model, 'r') as model_file:
        parameters = pickle.load(model_file)
    lasagne.layers.set_all_param_values(layer=network, values=parameters)

def save_network_as_model(network, model):
    parent_directory = os.path.abspath(model + "/../")
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
    parameters = lasagne.layers.get_all_param_values(layer=network)
    with open(model, 'w') as model_file:
        pickle.dump(parameters, model_file)

def preprocess(data):
    return data / numpy.float32(256)

def load_batch(batch_file):
    with open(batch_file, mode='rb') as opened_file:
        batch = pickle.load(opened_file)
        labels = batch[b'labels']
        datas = batch[b'data']
        names = batch[b'filenames']
    return names, datas, labels

def load_train_samples():
    number_of_labels = len(labels)
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    train_batch_files = [os.path.join(dataset, train_batch) for train_batch in train_batches]
    x_train = []; y_train = []
    for train_batch_file in train_batch_files:
        _, datas, labels = load_batch(train_batch_file)
        number_of_batch_samples = len(datas)
        for index in range(number_of_batch_samples):
            data = preprocess(data=numpy.reshape(datas[index], image_shape))
            label = [1 if labels[index] == j else 0 for j in range(number_of_labels)]
            x_train.append(data); y_train.append(label)
    datas = numpy.array(x_train, dtype=numpy.float32)
    labels = numpy.array(y_train, dtype=numpy.int8)
    return datas, labels

def load_test_samples():
    number_of_labels = len(labels)
    test_batch = 'test_batch'
    test_batch_file = os.path.join(dataset_path, test_batch)
    x_test = []; y_test = []
    _, datas, labels = load_batch(test_batch_file)
    number_of_samples = len(datas)
    for index in range(number_of_samples):
        data = preprocess(data=numpy.reshape(datas[index], image_shape))
        label = [1 if labels[index] == j else 0 for j in range(number_of_labels)]
        x_test.append(data); y_test.append(label)
    datas = numpy.array(x_test, dtype=numpy.float32)
    labels = numpy.array(y_test, dtype=numpy.int8)
    return datas, labels


def generate_batches():
    number_of_samples = len(datas)
    number_of_batch = number_of_samples / batch_size
    data_batches = numpy.split(datas, number_of_batch)
    label_batches = numpy.split(labels, number_of_batch)
    batches = [dict(data=data_batches[index], label=label_batches[index]) for index in range(number_of_batch)]
    return batches

def train():
    epoch_path = os.path.join(model_path, 'epochs')
    tensors = dict(input=theano.tensor.tensor4(dtype='float32'), output=theano.tensor.matrix(dtype='int8'))
    network = create_network(variable=tensors['input'])
    predictions = lasagne.layers.get_output(layer_or_layers=network)
    losses = loss(predictions=predictions, targets=tensors['output']).mean()
    parameters = lasagne.layers.get_all_params(layer=network, trainable=True)
    updates = updater(loss_or_grads=losses, params=parameters, learning_rate=rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    trainer = theano.function(inputs=[tensors['input'], tensors['output']], outputs=losses, updates=updates)
    batches = generate_batches(datas=datas, labels=labels)
    for epoch in range(epochs):
        print('Epoch {e}:'.format(e=(epoch+1)))
        number_of_batch = len(batches)
        for batch_index in range(number_of_batch):
            batch = batches[batch_index]
            batch_loss = trainer(batch['data'], batch['label'])
            print('Batch {b}: Loss = {l:.5f}'.format(b=(batch_index+1), l=batch_loss))
        epoch_file = 'epoch_{e}.params'.format(e=(epoch+1))
        epoch_model = os.path.join(epoch_path, epoch_file)
        save_network_as_model(network, epoch_model)
    trained_model_file = os.path.join(model_path, model)
    save_network_as_model(network, trained_model_file)


def predict():
    input_tensor = theano.tensor.tensor4(dtype='float32')
    network = create_network(variable=input_tensor)
    load_network_from_model(network=network, model=model)
    prediction = lasagne.layers.get_output(layer_or_layers=network, deterministic=True)
    result = theano.tensor.argmax(prediction, axis=1)
    predictor = theano.function(inputs=[input_tensor], outputs=result)
    if data_or_datas.shape != image_shape:
        datas = data_or_datas
        predictions = predictor(datas)
        return predictions
    else:
        channel, height, width = image_shape
        data = numpy.reshape(data_or_datas, newshape=(1, channel, height, width))
        prediction = predictor(data)
        return prediction

def test():
    number_of_samples = len(datas)
    predictions = predict(data_or_datas=datas, model=model)
    accurancy = 0
    for index in range(number_of_samples):
        prediction = predictions[index]
        target = numpy.argmax(labels[index])
        if target == prediction:
            accurancy += 1
    accurancy = (numpy.float32(accurancy) / number_of_samples) * 100
    print('Accurancy: {a:.3f}'.format(a=accurancy))

def main():
    print('Train samples are loading.')
    train_datas, train_labels = load_train_samples()
    print('Train samples are loaded.')
    print('Training:')
    train(datas=train_datas, labels=train_labels)
    print('Trained:')
    print('Test samples are loading.')
    test_datas, test_labels = load_test_samples()
    print('Testing:')
    test(datas=test_datas, labels=test_labels)
    print('Tested:')

#######

                
@app.route('/view_photo1',methods=['POST','GET'])
def view_photo1():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo1.html', result=value,vid=vid)

@app.route('/view_photo2',methods=['POST','GET'])
def view_photo2():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo2.html', result=value,vid=vid)    

@app.route('/view_photo3',methods=['POST','GET'])
def view_photo3():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo3.html', result=value,vid=vid)

@app.route('/view_photo4',methods=['POST','GET'])
def view_photo4():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo4.html', result=value,vid=vid)

@app.route('/message',methods=['POST','GET'])
def message():
    vid=""
    name=""
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT name FROM register where id=%s",(vid, ))
        name = mycursor.fetchone()[0]
    return render_template('message.html',vid=vid,name=name)


@app.route('/login',methods=['POST','GET'])
def login():
    uname=""
##    value=["1","2","3","4","5","6","7","8","9","0"]
##    change=random.shuffle(value)
##    print(change)
    if 'username' in session:
        uname = session['username']
    print(uname)
    mycursor1 = mydb.cursor()

    mycursor1.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor1.fetchone()
    accno=value[5]
    session['accno'] = accno
    
    mycursor1.execute("SELECT number FROM numbers order by rand()")
    value = mycursor1.fetchall()
    msg=""
        
    if request.method == 'POST':
        password1 = request.form['password']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where card=%s && password=%s",(uname, password1))
        myresult = mycursor.fetchone()[0]
        if password1=="":
            
            return render_template('login.html')
        else:
            
            #if str(password1)==str(myresult[10]):
            if myresult>0:
                #ff2=open("log.txt","w")
                #ff2.write(password1)
                #ff2.close()
                result=" Your Logged in sucessfully**"
                
                return redirect(url_for('userhome'))
            else:
                msg="Your logged in fail!!!"
                #return render_template('userhome.html',result=result)
    
    
    return render_template('login.html',value=value,msg=msg)



@app.route('/userhome')
def userhome():
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close() 

    name=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor.fetchone()
    

    print(uname)
    
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor.fetchone()
    print(value)
    name=value[1]  
        
    return render_template('userhome.html',name=name,value=value)

'''@app.route('/deposit')
def deposit():
    return render_template('deposit.html')
@app.route('/deposit_amount',methods=['POST','GET'])
def deposit_amount():
    if request.method=='POST':
        name=request.form['name']
        accountno=request.form['accno']
        amount=request.form['amount']
        today = date.today()
        rdate = today.strftime("%b-%d-%Y")
        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM event")
        maxid = mycursor.fetchone()[0]
        sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
        val = (maxid, name, accountno, amount, rdate)
        mycursor.execute(sql, val)
        mydb.commit()   
    return render_template('userhome.html')'''

'''@app.route('/withdraw')
def withdraw():

    
    return render_template('withdraw.html')'''

@app.route('/verify_face',methods=['POST','GET'])
def verify_face():
    msg=""
    ss=""
    uname=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    
                
    return render_template('verify_face.html',msg=msg)

@app.route('/face',methods=['POST','GET'])
def face():
    msg=""
    ss=""
    uname=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    print("uname="+uname)
    shutil.copy('static/faces/f1.jpg', 'static/f1.jpg')

    ff3=open("img.txt","r")
    mcnt=ff3.read()
    ff3.close()

    mcnt1=int(mcnt)
    if mcnt1==2:
        msg="Face Detected"
    elif mcnt1>2:
        msg="Multiple Face Detected!"
    else:
        msg=""
   
    
    
                
    return render_template('face.html',msg=msg,act=act,mcnt1=mcnt1)

@app.route('/process',methods=['POST','GET'])
def process():
    vid=""
    pg="0"
    act="1"
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    value=[]
    shutil.copy('static/faces/f1.jpg', 'static/f1.jpg')
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()
    name=account[1]
    mobile=account[3]
    
    email=account[4]
    vid=account[0]
    cursor.execute("SELECT vface FROM vt_face where vid=%s limit 0,1",(vid, ))
    value = cursor.fetchone()[0]
        
    
    return render_template('process.html', vid=vid,pg=pg,act=act,result=value)

@app.route('/pro',methods=['POST','GET'])
def pro():
    vid=""
    value=[]
    pgg=0
    act="1"
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    if request.method=='GET':
        act = request.args.get('act')
    
        vid = request.args.get('vid')
        pg = request.args.get('pg')
        #pgg=int(pg)+1
        pgg=2
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(vid,))
        dtt = mycursor.fetchone()[0]
        
        if dtt<=pgg:
            act="1"
        else:
            act="2"
        
        mycursor.execute("SELECT vface FROM vt_face where vid=%s limit 0,1",(vid, ))
        value = mycursor.fetchone()[0]
        #print(value)
        
    return render_template('pro.html', result=value,vid=vid,pg=pgg,act=act)

@app.route('/verify_face2',methods=['POST','GET'])
def verify_face2():
    msg=""
    ss=""
    uname=""
    mess=""
    act=""
    if request.method=='GET':
        act = request.args.get('act')
        
    #if 'username' in session:
    #    uname = session['username']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()

    ff2=open("bc.txt","r")
    bc=ff2.read()
    ff2.close()

    ff2=open("facest.txt","r")
    fst=ff2.read()
    ff2.close()
    
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()
    id1=str(account[0])
    name=account[1]
    mobile=account[3]
    print(mobile)
    email=account[4]
    vid=account[0]
    
    
    shutil.copy('static/faces/f1.jpg', 'faces/s1.jpg')
    cutoff=5
    img="v"+str(vid)+".jpg"
    '''cursor.execute('SELECT * FROM vt_face WHERE vid = %s', (vid, ))
    dt = cursor.fetchall()
    for rr in dt:
        hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
        hash1 = imagehash.average_hash(Image.open("faces/s1.jpg"))
        cc1=hash0 - hash1
        print("cc="+str(cc1))
        if cc1<=cutoff:
            ss="ok"
            break
        else:
            ss="no"'''

    
    if id1==fst:
        act="2"
        msg="Face Verified"
        print("correct person")
        return redirect(url_for('userhome', msg=msg))
    else:
        act="1"
        msg="Unknown Face Found"
        print("wrong person")
        #xn=randint(1000, 9999)
        #otp=str(xn)
        
        #cursor1 = mydb.cursor()
        #cursor1.execute('update register set otp=%s WHERE card = %s', (otp, uname))
        #mydb.commit()

        mess="Someone Access your account"
        url2="http://localhost/atm1/img.txt"
        ur = urlopen(url2)#open url
        data1 = ur.read().decode('utf-8')

       
        #idd=int(data1)+1
        #url="http://iotcloud.co.in/testsms/sms.php?sms=link12&name="+name+"&mess="+mess+"&mobile="+str(mobile)+"&bc="+bc
        #print(url)
        #webbrowser.open_new(url)
            
                
    return render_template('verify_face2.html',msg=msg,act=act,mess=mess,mobile=mobile,name=name,bc=bc)

@app.route('/cap',methods=['POST','GET'])
def cap():
    msg=""

    ff2=open("bc.txt","r")
    bc=ff2.read()
    ff2.close()

    
    
    return render_template('cap.html',msg=msg,bc=bc)

@app.route('/verify',methods=['POST','GET'])
def verify():
    msg=""
    data1=""
    #act=""
    amtt=""
    cc=""
    name=""
    mobile=""
    mess=""
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    #data1="4"

    ff2=open("bc.txt","r")
    bc=ff2.read()
    ff2.close()

    logfn=bc+".txt"
    
    url2="http://localhost/atm1/"+logfn
    ur = urlopen(url2)#open url
    data1 = ur.read().decode('utf-8')
    vv=data1.split('-')
    data1=vv[0]
    amtt=vv[1]
    print(data1)
    
    act = request.args.get('act')
    if act is None:
        act=""
    
    print("act="+str(act))
    if data1=="accept":
        act="1"

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor.fetchone()
  
    
    if act=="3":
        amt=0
        amt1=0
        amt2=0
    
        
        amount1=amtt
        
       
        mycursor.execute("SELECT amount FROM admin where username='admin'")
        amt1 = mycursor.fetchone()[0]

        mycursor.execute("SELECT deposit FROM register where card=%s",(uname, ))
        amt2 = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
        ddt = mycursor.fetchone()
        name=ddt[1]
        mobile=ddt[3]

        amt=int(amount1)
        if amt<=amt1:

            if amt<=amt2:
                #mycursor.execute("UPDATE admin SET amount=amount-%s WHERE username='admin'",(amount1, ))
                #mydb.commit()
                mycursor.execute("UPDATE register SET deposit=deposit-%s WHERE card=%s",(amount1, uname))
                mydb.commit()

                now = datetime.datetime.now()
                rdate=now.strftime("%d-%m-%Y")
                mycursor.execute("SELECT max(id)+1 FROM event")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
                val = (maxid, name, uname, amt, rdate)
                mycursor.execute(sql, val)
                mydb.commit()

                mess="Amount Debited Rs."+str(amt)
                url="http://iotcloud.co.in/testsms/sms.php?sms=msg&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                webbrowser.open_new(url)
            
                msg="Withdraw success..."
            else:
                mess="Your Account balance is low!"
                #url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                #webbrowser.open_new(url)
                msg="Your Account balance is low!"
        else:
            msg="Cash is not available in ATM!!"
    
        
    return render_template('verify.html',msg=msg,act=act,amtt=amtt,data1=data1,name=name,mobile=mobile,mess=mess,value=value)


@app.route('/otp', methods=['GET', 'POST'])
def otp():
    msg=""
    key=""
    if 'username' in session:
        uname = session['username']
    cursor = mydb.cursor()
    cursor.execute('SELECT otp FROM register WHERE card = %s', (uname, ))
    account = cursor.fetchone()[0]
    key=account
    
    if request.method=='POST':
        otp=request.form['otp']
        
        if otp==key:
            session['username'] = uname
            
            return redirect(url_for('verify_aadhar'))
        else:
            msg = 'OTP wrong!'
    return render_template('otp.html',msg=msg,key=key)

@app.route('/atm_balance',methods=['POST','GET'])
def atm_balance():
    msg=""
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()

    cursor = mydb.cursor()
    if request.method=='POST':
        amount=request.form['amount']
        cursor.execute("UPDATE admin SET amount=%s WHERE username='admin'",(amount, ))
        mydb.commit()
        return redirect(url_for('admin'))

        
    
    cursor.execute("SELECT amount FROM admin WHERE username='admin'")
    value = cursor.fetchone()[0]
    
    return render_template('atm_balance.html',msg=msg,value=value)

@app.route('/withdraw',methods=['POST','GET'])
def withdraw():
    uname=""
    ##if 'username' in session:
    #    uname = session['username']
    #    accno = session['accno']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor.fetchone()
    st="" 
    msg=""
    amt=0
    amt1=0
    amt2=0
    name=""
    mobile=""
    mess=""
    if request.method=='POST':
        
        amount1=request.form['amount']
        
        

        mycursor.execute("SELECT amount FROM admin where username='admin'")
        amt1 = mycursor.fetchone()[0]

        mycursor.execute("SELECT deposit FROM register where card=%s",(uname, ))
        amt2 = mycursor.fetchone()[0]

        mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
        ddt = mycursor.fetchone()
        name=ddt[1]
        mobile=ddt[3]

        amt=int(amount1)
        if amt<=amt1:

            if amt<=amt2:
                #mycursor.execute("UPDATE admin SET amount=amount-%s WHERE username='admin'",(amount1, ))
                #mydb.commit()
                mycursor.execute("UPDATE register SET deposit=deposit-%s WHERE card=%s",(amount1, uname))
                mydb.commit()

                now = datetime.datetime.now()
                rdate=now.strftime("%d-%m-%Y")
                mycursor.execute("SELECT max(id)+1 FROM event")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1
                sql = "INSERT INTO event(id, name, accno, amount, rdate) VALUES (%s, %s, %s, %s, %s)"
                val = (maxid, name, uname, amt, rdate)
                mycursor.execute(sql, val)
                mydb.commit()

                mess="Amount Debited Rs."+str(amt)
                #url="http://iotcloud.co.in/testsms/sms.php?sms=emr&name="+name+"&mess="+mess+"&mobile="+str(mobile)
                #webbrowser.open_new(url)
                st="1"
                msg="Withdraw success..."
            else:
                st="2"
                msg="Your Account balance is low!"
        else:
            st="3"
            msg="Cash is not available in ATM!!"
        
    return render_template('withdraw.html',msg=msg,name=name,mobile=mobile,mess=mess,value=value,st=st)


@app.route('/balance')
def balance():
    uname=""
    #if 'username' in session:
    #    uname = session['username']
    #    accno = session['accno']
    ff2=open("un.txt","r")
    uname=ff2.read()
    ff2.close()
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    value = mycursor.fetchone()
    
    
    mycursor.execute("SELECT * FROM register where card=%s",(uname, ))
    data = mycursor.fetchone()
    deposit=data[9]
    print(str(deposit))
    return render_template('balance.html', data=deposit,value=value)



@app.route('/user_view')
def user_view():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    result = mycursor.fetchall()
    return render_template('user_view.html', result=result)

@app.route('/view_withdraw')
def view_withdraw():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM event order by id desc")
    result = mycursor.fetchall()
    return render_template('view_withdraw.html', result=result)

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))
#########
def gen2(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed2')       
def video_feed2():
    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
################
def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
