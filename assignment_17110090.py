import os
import cv2
import numpy as np
from PIL import Image
from numpy import linalg as LA
rows,columns=400,400
def facecrop(i):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)
    image=Image.open('YaleFaceDatabase/'+i).convert('L')
    img=np.array(image,'uint8')
    img= cv2.merge([img,img,img])
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)
    
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
        gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (rows,columns), interpolation = cv2.INTER_AREA)
        k=i.replace('.','_')
        cv2.imwrite('images_gray/'+k+'.jpg', resized)
    return
images= os.listdir('./YaleFaceDatabase')
all_images=[]
for i in images:
    facecrop(i)
print('filtered images')
sample_image=cv2.imread('images_gray/subject15_happy.jpg',0)
print('dimensions : ',sample_image.shape)
print('type : ','gray')
cv2.imshow('Sample',sample_image)
cv2.waitKey()
images= os.listdir('./images_gray')
all_names=[]
all_images=[]
# 8 out of 11 of each
test_names=[]
test_images=[]
# 3 out of 11 of each

#testsplit= 73 : 27
for i in range(len(images)):
    k=cv2.imread('images_gray/'+images[i],0)
    arr=np.array(k)
    arr1=np.reshape(arr,(1,-1))
    if i%11<=7:
        all_images.append(arr1)
        all_names.append(images[i])
    else:
        test_images.append(arr1)
        test_names.append(images[i])
l=np.mean( np.array(all_images), axis=0)
mean_of_imgs=l
cv2.imshow('Mean',np.array(np.reshape(l,(rows,-1)),dtype = np.uint8))
cv2.waitKey()
all_im_mean=[]
for i  in all_images:
	all_im_mean.append((i-l)[0])
all_im_mean=np.array(all_im_mean)
A=np.transpose(all_im_mean)
A_t=np.transpose(A)
cov=np.matmul(A_t,A)
#print(cov)
w, v = LA.eig(cov)
di=[[] for i in range(len(w))]
k1=np.array(sorted(w)[::-1])
for i in range(len(w)):
	di[i]=[w[i],v[i].tolist()]
k5=k1/sum(k1)
n=0
s=0
while(s<0.85):
    s+=k5[n]
    n+=1
top=31
req_vec=[]
req_eigens=sorted(di)[::-1][:top]
import cmath
for i in req_eigens:
    lr=[]
    for j in i[1]:
        lr.append(float(j.real))
    req_vec.append(lr)
req_eigens=np.array(req_vec).transpose()
actual_eigens=np.matmul(A,req_eigens)
norm_=np.linalg.norm(actual_eigens,axis=0)
actual_eigens=actual_eigens/norm_
actual_eigens=actual_eigens.T
all_weights=np.matmul(actual_eigens,A)
all_weights=all_weights.T
all_trans_imgs=[]
c1=0
for i in range(len(all_images)):
    c1+=1
    s=[0 for m in range(rows*columns)]
    for j in range(len(actual_eigens)):
        s=np.add(np.array(s),np.array(actual_eigens[j])*all_weights[i][j])
    l=l.T
    l=l.flatten()
    s=s+l
    if c1<=3:
        cv2.imshow('actual_image',np.array(np.reshape(np.array(all_images[i]),(rows,-1)),dtype = np.uint8))
        cv2.imshow('projection',np.array(np.reshape(np.array(s),(rows,-1)),dtype = np.uint8))
        cv2.waitKey()
    all_trans_imgs.append(s)
           
test_img_centered=[]
for i in test_images:
	test_img_centered.append((i-l)[0])
test_without_mean=np.array(test_img_centered)
A_test=np.transpose(test_without_mean)
test_weights=np.matmul(actual_eigens,A_test)
test_weights=test_weights.T
correct=0
wrong=0
threshould=20000
for i in range(len(test_weights)):
    distances=[]
    for j in all_weights:
        distances.append(np.linalg.norm(j-test_weights[i]))
    predicted_name=all_names[np.argmin(distances)]
    predicted_name=predicted_name[:predicted_name.find('_')]
    actual_name=test_names[i][:test_names[i].find('_')]
    if (predicted_name==actual_name) and (min(distances)<threshould):
        correct+=1
    else:
        wrong+=1
    print('actual_name: ',actual_name,' | ','predicted_name: ',predicted_name)
print('threshould: ',threshould)
print('____________________________________________________________')
print('accuracy: ',(correct/(correct+wrong))*100)

