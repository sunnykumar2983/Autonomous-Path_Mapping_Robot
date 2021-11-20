import numpy as np
import cv2
import math
import cv2.aruco as aruco
import serial
import time
#########
n=9
time.sleep(0.5)
ser=serial.Serial('COM6',baudrate=9600,timeout=1)
cap=cv2.VideoCapture(1)
matrix1= np.full([n*n,n*n],10000,dtype=int)
_,img=cap.read()

roi=cv2.selectROI(img)
img2=img[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
shape=np.zeros([n,n])


height=img2.shape[0]//n
width=img2.shape[1]//n

color_info=[]
for i in range(n*n):
    if(i+n<n*n):
        matrix1[i][i+n]=10
        matrix1[i+n][i]=10
    if(((i+1)//n)*n-i!=1):
        matrix1[i][i+1]=10
        matrix1[i+1][i]=10
    
matrix=matrix1
def bot_pos():
    ret,aruco_img=cap.read()
    #aruco_img=cv2.resize(img,(512,512))
    aruco_img=aruco_img[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
    #cv2.imshow("aruco image ",aruco_img)
    #aruco_img=aruco_img[y:y+h,x:x+w]
    #aruco_img=cv2.resize(aruco_img,(0,0),fx=0.25,fy=0.25)
    gray=cv2.cvtColor(aruco_img,cv2.COLOR_BGR2GRAY)
    aruco_dict=aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters=aruco.DetectorParameters_create()
    corners,ids,_=aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
    if(ids==None):
        print("not detected")
        '''ser.write(b'b')#........................
        time.sleep(0.2)
        ser.write(b'f')#..................
        time.sleep(0.18)
        #cap.clear()
        #cap=cv2.VideoCapture(1)'''
        return [-1,-1],-1j
    #print(corners)
    corners=np.array(corners,dtype=int)
    bot_cen_x_y=[(corners[0][0][0][0]+corners[0][0][1][0]+corners[0][0][2][0]+corners[0][0][3][0])/4,(corners[0][0][0][1]+corners[0][0][1][1]+corners[0][0][2][1]+corners[0][0][3][1])/4]
    #print("bot coordinates: ",bot_cen_x_y)
    vector=complex(corners[0][0][0][0]-corners[0][0][3][0],corners[0][0][0][1]-corners[0][0][3][1])
    #print("vector: ",vector)
    #print("corners list:",corners)
    return bot_cen_x_y,vector
def movement(path):
    for i in path:
        print("current node is: ",i)
        bot_cen_x_y,z_bot=bot_pos()
        while(bot_cen_x_y==[-1,-1] and z_bot==-1j):
            bot_cen_x_y,z_bot=bot_pos()
        next_node_x=width*(i%n)+width/2
        next_node_y=height*(i//n)+height/2;
        #print("next_node cord: ",next_node_x," ",next_node_y)
        z_dest=complex(next_node_x-bot_cen_x_y[0],next_node_y-bot_cen_x_y[1])
        #print("Z_bot: ",z_bot)
        #print("Z_dest: ",z_dest)
        to_rotate=np.angle(z_bot/z_dest,True)
        print("angle to rotate",to_rotate)
        distance=math.sqrt(((next_node_x-bot_cen_x_y[0])*(25/width))**2+((next_node_y-bot_cen_x_y[1])*(25/height))**2)
        print("distance to cover: ",distance)
        thresh=5
        if(i==path[-1]):
            thresh=18
            while(distance>thresh):
                if(to_rotate>=-10 and to_rotate<=10):
                    print('f')
                    ser.write(b'f')
                    time.sleep(0.20)
                    ser.write(b's')
                    
                elif(to_rotate>10):
                    print('l')
                    ser.write(b'l')
                    time.sleep(0.19)
                    
                elif(to_rotate<10):
                    print('r')
                    ser.write(b'r')
                    time.sleep(0.19)
                    ser.write(b's')
                   
                        
                    
                bot_cen_x_y,z_bot=bot_pos()
                
                z_dest=complex(next_node_x-bot_cen_x_y[0],next_node_y-bot_cen_x_y[1])
                
                to_rotate=np.angle(z_bot/z_dest,True)
                distance=math.sqrt(((next_node_x-bot_cen_x_y[0])*(25/width))**2+((next_node_y-bot_cen_x_y[1])*(25/height))**2)


        else:
            while(distance>thresh):
                if(to_rotate>=-23 and to_rotate<=23):
                    print('f')
                    ser.write(b'f')
                    time.sleep(distance/(37))
                    ser.write(b's')
                elif to_rotate>23:
                    print('l')
                    ser.write(b'l') 
                    time.sleep(abs(to_rotate)*0.0030)
                    ser.write(b's')
                elif to_rotate<-23:
                    print('r')
                    ser.write(b'r')
                    time.sleep(abs(to_rotate)*0.0030)
                    ser.write(b's')
                        
                    
                bot_cen_x_y,z_bot=bot_pos()
                
                z_dest=complex(next_node_x-bot_cen_x_y[0],next_node_y-bot_cen_x_y[1])
                
                to_rotate=np.angle(z_bot/z_dest,True)
                distance=math.sqrt(((next_node_x-bot_cen_x_y[0])*(25/width))**2+((next_node_y-bot_cen_x_y[1])*(25/height))**2)

#for red..................................................
roi0=cv2.selectROI(img2)
colorrange=img2[int(roi0[1]):int(roi0[1]+roi0[3]),int(roi0[0]):int(roi0[0]+roi0[2])]
'''rmin=colorrange[:,:,2].min()
rmax=colorrange[:,:,2].max()
gmin=colorrange[:,:,1].min()
gmax=colorrange[:,:,1].max()
bmin=colorrange[:,:,0].min()
bmax=colorrange[:,:,0].max()'''
rmean=np.mean((colorrange[:,:,2]))
gmean=np.mean((colorrange[:,:,1]))
bmean=np.mean((colorrange[:,:,0]))
#print(bmin," ",bmax," ",gmin," ",gmax," ",rmin," ",rmax," ")
print("red: ",bmean," ",gmean," ",rmean)
#lowerc=np.array([bmin-35,gmin-38,rmin-38])
#upperc=np.array([bmax+35,gmax+35,rmax+35])
lowerc=np.array([bmean-55,gmean-55,rmean-55])
upperc=np.array([bmean+55,gmean+55,rmean+55])
'''lowerc=np.array([bmean-75,gmean-95,rmean-95])
upperc=np.array([bmean+95,gmean+130,rmean+140])'''

x=[]
x.append(list(lowerc))
x.append(list(upperc))
color_info.append(x)
mask=cv2.inRange(img2,lowerc,upperc)
mask = cv2.dilate(mask,np.ones((2,2)),iterations=2)
_,contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.imshow("mask",mask)
#cv2.waitKey(1)
font=cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    if cv2.contourArea(cnt)>250:
        approx=cv2.approxPolyDP(cnt,0.028*cv2.arcLength(cnt,True),True)
        #cv2.drawContours(img2,[approx], -1, (0, 255, 0), 3)
        if len(approx)==4:
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape[cY//height][cX//width]=3
        elif len(approx)>4:
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape[cY//height][cX//width]=1;

#FOR YELLOW.........................................................................
roi0=cv2.selectROI(img2)
colorrange=img2[int(roi0[1]):int(roi0[1]+roi0[3]),int(roi0[0]):int(roi0[0]+roi0[2])]

rmean=np.mean((colorrange[:,:,2]))
gmean=np.mean((colorrange[:,:,1]))
bmean=np.mean((colorrange[:,:,0]))

print(bmean," ",gmean," ",rmean)

lowerc=np.array([bmean-75,gmean-95,rmean-95])
upperc=np.array([bmean+95,gmean+150,rmean+130])
x=[]
x.append(list(lowerc))
x.append(list(upperc))
color_info.append(x)
mask=cv2.inRange(img2,lowerc,upperc)
mask = cv2.dilate(mask,np.ones((2,2)),iterations=2)
_,contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.imshow("mask",mask)
#cv2.waitKey(1)
font=cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    if cv2.contourArea(cnt)>150:
        approx=cv2.approxPolyDP(cnt,0.028*cv2.arcLength(cnt,True),True)
        
        if len(approx)==4:
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape[cY//height][cX//width]=4
        elif len(approx)>4:
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape[cY//height][cX//width]=2;

#FOR WHITE.....................................................
roi0=cv2.selectROI(img2)
colorrange=img2[int(roi0[1]):int(roi0[1]+roi0[3]),int(roi0[0]):int(roi0[0]+roi0[2])]

rmean=np.mean((colorrange[:,:,2]))
gmean=np.mean((colorrange[:,:,1]))

print(bmean," ",gmean," ",rmean)

lowerc=np.array([bmean-35,gmean-35,rmean-35])
upperc=np.array([bmean+45,gmean+40,rmean+40])
x=[]
x.append(list(lowerc))
x.append(list(upperc))
color_info.append(x)
mask=cv2.inRange(img2,lowerc,upperc)
mask = cv2.dilate(mask,np.ones((2,2)),iterations=2)
_,contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.imshow("mask",mask)
#cv2.waitKey(1)
font=cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    if cv2.contourArea(cnt)>150:
        approx=cv2.approxPolyDP(cnt,0.028*cv2.arcLength(cnt,True),True)
        
    
        if len(approx)==4:
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            shape[cY//height][cX//width]=5



#FOR BLUE................................

roi0=cv2.selectROI(img2)
colorrange=img2[int(roi0[1]):int(roi0[1]+roi0[3]),int(roi0[0]):int(roi0[0]+roi0[2])]
'''rmin=colorrange[:,:,2].min()
rmax=colorrange[:,:,2].max()
gmin=colorrange[:,:,1].min()
gmax=colorrange[:,:,1].max()
bmin=colorrange[:,:,0].min()
bmax=colorrange[:,:,0].max()'''
rmean=np.mean((colorrange[:,:,2]))
gmean=np.mean((colorrange[:,:,1]))
bmean=np.mean((colorrange[:,:,0]))

print("for blue: ",bmean," ",gmean," ",rmean)

lowerc=np.array([bmean-25,gmean-25,rmean-25])
upperc=np.array([bmean+85,gmean+85,rmean+85])
x=[]
x.append(list(lowerc))
x.append(list(upperc))
color_info.append(x)
mask=cv2.inRange(img2,lowerc,upperc)
mask = cv2.dilate(mask,np.ones((2,2)),iterations=2)
_,contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.imshow("mask",mask)
#cv2.waitKey(1)
font=cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    if cv2.contourArea(cnt)>350:
        print(cv2.contourArea(cnt))
        approx=cv2.approxPolyDP(cnt,0.028*cv2.arcLength(cnt,True),True)
        x,y,w,h = cv2.boundingRect(cnt)
    
        if len(approx)==4:
            x=approx.ravel()[0]
            y=approx.ravel()[1]
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("contour area for blue: ",cv2.contourArea(cnt))
            if(shape[cY//height][cX//width]!=0):
               shape[cY//height][cX//width]=-1
            else:
                shape[cY//height][cX//width]=6
            if(cv2.contourArea(cnt)>10000):
                if(shape[cY//height][(cX+width)//width]!=0):
                    shape[cY//height][(cX+width)//width]=-1
                else:
                   shape[cY//height][(cX+width)//width]=6
                if(shape[cY//height][(cX-width)//width]!=0):
                    shape[cY//height][(cX-width)//width]=-1
                else:
                   shape[cY//height][(cX-width)//width]=6
                        
            
            
#FOR GREEN..............................
print("FOR GREEN")
green=[]
#cv2.imshow("img2",img2)
cv2.waitKey(0)
roi0=cv2.selectROI(img2)
colorrange=img2[int(roi0[1]):int(roi0[1]+roi0[3]),int(roi0[0]):int(roi0[0]+roi0[2])]
'''rmin=colorrange[:,:,2].min()
rmax=colorrange[:,:,2].max()
gmin=colorrange[:,:,1].min()
gmax=colorrange[:,:,1].max()
bmin=colorrange[:,:,0].min()
bmax=colorrange[:,:,0].max()'''
rmean=np.mean((colorrange[:,:,2]))
gmean=np.mean((colorrange[:,:,1]))
bmean=np.mean((colorrange[:,:,0]))

print("FOR GREEN ",bmean," ",gmean," ",rmean)
lowerc=np.array([bmean-50,gmean-45,rmean-30])
upperc=np.array([bmean+55,gmean+60,rmean+50])




mask=cv2.inRange(img2,lowerc,upperc)
mask = cv2.erode(mask,np.ones((2,2)),iterations=2)
_,contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.imshow("mask",mask)
cv2.waitKey(1)
font=cv2.FONT_HERSHEY_COMPLEX
for cnt in contours:
    if cv2.contourArea(cnt)>500:
        approx=cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        
        M = cv2.moments(approx)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        green.append((cY//height)*n+cX//width)
        
print("positiion of green: ",green)
shape[77//n][77%n]=-1
shape[76//n][76%n]=-1
shape[75//n][75%n]=6
white_corner=[]
for i in green:
    if(shape[i//n][i%n]==5):
        white_corner.append(i)

print("white with green corner :",white_corner)
#disconnection
white_lower=np.array(color_info[2][0])
white_upper=np.array(color_info[2][1])
def disconnect(matrix,leave_node):
    _,img0=cap.read()
    #img0=cv2.resize(img,(512,512))
    img0=img0[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
    mask0=cv2.inRange(img0,white_lower,white_upper)
    mask0 = cv2.dilate(mask0,np.ones((2,2)),iterations=2)
    _,contours,_= cv2.findContours(mask0,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    #cv2.imshow("mask",mask0)
    #cv2.waitKey(0)
    font=cv2.FONT_HERSHEY_COMPLEX
    for cnt in contours:
        if cv2.contourArea(cnt)>250:
            approx=cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt,True),True)
            #cv2.drawContours(img0,[approx], -1, (0, 255, 0), 3)
            #print("moment is calculating")
            M = cv2.moments(approx)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            node=(cX//width)+(cY//height)*n
            #or node==leave_node+1 or node==leave_node+5 or node==leave_node-1 or node==leave_node-5
            if(node==leave_node):
                continue
            print("node to disconnect: ",node)
            for i in range(n*n):
                if(matrix[node][i]!=10000):
                    #print("nearby node:",i)
                    matrix[node][i]=10000
                    matrix[i][node]=10000        
    return
def disconnect_target(target):
    for i in range(n*n):
        if(matrix[target][i]!=10000):
            print("target and disconnected node for target is:",target," ",i)
            matrix[target][i]=10000
            matrix[i][target]=10000

            
def connect_node(node,matrix):
    print("connect node:",node)
    if(node+n<n*n):
        matrix[node][node+n]=10
        matrix[node+n][node]=10
    if(((node+1)//n)*n-node!=1):
        matrix[node][node+1]=10
        matrix[node+1][node]=10
    if(node-n>-1):
        matrix[node][node-n]=10
        matrix[node-n][node]=10
    if(node%n!=0):
        matrix[node][node-1]=10
        matrix[node-1][node]=10

#a=[]
def MinDistanceNode(visited,distance,nodes):
    minimum=1000
    minnode=0
    for i in range(nodes):
        if(visited[i]==False and distance[i]<=minimum):
            minimum=distance[i]
            minnode=i
    return minnode

    
def dijkstras(matrix,src,target,nodes,visited,distance):
    distance[src]=0
    parent=np.zeros(nodes,dtype=int)
    for i in range(nodes):
        parent[i]=-1;
    if(target==src):
        print("target and src is same")
        parent[target]=src
    for node in range(nodes):
        currentnode=MinDistanceNode(visited,distance,nodes)
        visited[currentnode]=True
        if(visited[target]==True):
            return parent
        for neighbour in range(nodes):                                    #///////
            if(visited[neighbour]==False and matrix[currentnode][neighbour]!=10000 and distance[currentnode]+matrix[currentnode][neighbour]<distance[neighbour]):
                distance[neighbour]=distance[currentnode]+matrix[currentnode][neighbour]
                parent[neighbour]=currentnode
    
def revealed(node):
    time.sleep(0.3)##################
    i=node//n
    j=node%n 
    _,img0=cap.read()
    #img0=cv2.resize(img,(512,512))
    img0=img0[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]
    cv2.imshow("image revealing",img0)
    cv2.waitKey(1)
    cropped=img0[i*height:(i+1)*height,j*width:(j+1)*width,:]
    for i in range(2):
        lowerc=np.array(color_info[i][0])
        upperc=np.array(color_info[i][1])
        mask=cv2.inRange(cropped,lowerc,upperc)
        mask = cv2.dilate(mask,np.ones((2,2)),iterations=2)
        cv2.imshow("mask for cropped",mask)
        cv2.waitKey(1)
        _,contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     
        for cnt in contours:
            if cv2.contourArea(cnt)>50:
                approx=cv2.approxPolyDP(cnt,0.028*cv2.arcLength(cnt,True),True)
                #cv2.drawContours(img2,[approx], -1, (0, 255, 0), 3)
                x,y,w,h = cv2.boundingRect(cnt)##################min bounding rectange can be used if box is rotated
                if(cv2.contourArea(cnt)/(w*h)>.82):
                    print("shape revealed is:",i+3)
                    shape[node//n][node%n]=i+3
                    return i+3
                        
                else:
                    print("shape revealed is:",i+1)
                    shape[node//n][node%n]=i+1
                    return i+1
    

destination=[]
#...take care
blue=[]
blue0=[]
print("final shape: ",shape)
for i in range(n*n):
    if(shape[i//n][i%n]==6):
        blue.append(i)
        blue0.append(i)

print("position of all blue",blue)
blue_left=[]
#select blue which will be engaged as jail in first run ,jail blocked from all side will not be available

for i in blue:
    s=[]
    for j in range(n*n):
        if(matrix[i][j]==10):#node j is connected to blue
            s.append(j)     #s gives the nodes that are connected to blue
    y=0
    for conn_to_blue in s:
        if(shape[conn_to_blue//n][conn_to_blue%n]==5 or shape[conn_to_blue//n][conn_to_blue%n]==-1):
            y=y+1
    if(y==len(s)):
        blue.remove(i)
        blue_left.append(i)
    
print("left jail is: ",blue_left)  
        
#blue0 will is unchangable one
print("blue0 and blue: ",blue0)
white_noncorner=[]
for i in range(n):
    for j in range(n):
        if(shape[i][j]==5 and ((i*n+j) not in white_corner)):
                white_noncorner.append(i*n+j)
                
startcord,_=bot_pos()
start=(startcord[0]//width)+(startcord[1]//height)*n
l,m=start//n,start%n
c=0
print("white_corners: ",white_corner)
print("white_noncorners: ",white_noncorner)
#all places of white
white_total=white_corner+white_noncorner
print("white all places:",white_total)
#green blue is used for disconnecting...if that is a target  
Green_blue=green+blue
print("nodes that will be blocked(green+blue) :",Green_blue)
#print("blue :",blue)
#choose the least distance white block
track_removed_wc=[]
while(len(white_corner)>0 and len(blue)>0):
        node=white_corner[0]
        x=1000
        for i in white_corner:
            #print("white corner node and its distance: ",abs(l-i//n)+abs(m-i%n))
            if(abs(l-i//n)+abs(m-i%n)<=x):
                x=abs(l-i//n)+abs(m-i%n)
                node=i
        print("least corner white: ",node)
        destination.append(node)
        l=node//n
        m=node%n
        white_corner.remove(node)
        track_removed_wc.append(node)
        node=blue[0]
        x=1000
        for i in blue:
            print("blue cell node and its distance: ",abs(l-i//n)+abs(m-i%n))
            if(abs(l-i//n)+abs(m-i%n)<=x):
                x=abs(l-i//n)+abs(m-i%n)
                node=i
        #print("blue position node: ",node)
        print("least prison: ",node)
        destination.append(node)
        l=node//n
        m=node%n
        blue.remove(node)

print("white at green nodes remaining after putting in jail:",white_corner)
print("destination :",destination)
print("start",start)
disconnect(matrix,start)      #### BOX FROM GREEN TO PRISON

print(shape)
cv2.waitKey(0)
for target in destination:
    print("blue0 isssssss",blue0)
    print("current target: ",target)
    connect_node(target,matrix)
    #print("matrix: ",matrix)
    src_cen,v=bot_pos()
    while(src_cen==[-1,-1] and v==-1j):
            src_cen,v=bot_pos()
    node_column=src_cen[0]//width
    node_row=src_cen[1]//height
    print("row :column",node_row,", ",node_column)
    src=int(node_row*n+node_column)
    print("source node and target is: ",int(src)," ",target)
    nodes=matrix.shape[0]
    visited=[False]*nodes
    distance=[1000]*nodes
    parent=dijkstras(matrix,src,target,nodes,visited,distance)
    #to remove target point.....
    path=[target]
    x=parent[target]
    while(x==-1):
        print("x is -1")
        print('b')
        ser.write(b'b')
        time.sleep(0.5)
        ser.write(b's')
        src_cen,v=bot_pos()
        while(src_cen==[-1,-1] and v==-1j):
                src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row :column",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        print("source node and target is: ",int(src)," ",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target]
        
    while(x!=src):
        print(" x is:",x)
        path.insert(0,x)
        x=parent[x]
    #path.insert(0,src)
    print("path to follow: ",path)
    movement(path)
    if(target in green):
        print("target is in green")
        ser.write(b'd')
        print('d')
        time.sleep(1.1)
        ser.write(b's')
        print('b')
        ser.write(b'b')
        time.sleep(0.3)
        ser.write(b's')
    else:
        print("blink blue")
        ser.write(b'B')
        time.sleep(0.5)
        ser.write(b'u')
        print('u')
        time.sleep(1.1)
        print('b')
        ser.write(b'b')
        time.sleep(0.9)
        ser.write(b's')

    src_cen0,v0=bot_pos()
    while(src_cen0==[-1,-1] and v0==-1j):
            src_cen0,v0=bot_pos()
    node_column0=src_cen0[0]//width
    node_row0=src_cen0[1]//height
    print("row :column0",node_row0,", ",node_column0)
    src0=int(node_row0*n+node_column0)
    print("source node is: ",int(src0))
    #destination achieved if target was in blue box disconnect that target/box
    print("blue0 is: and target is",blue0," ",target)
    if(target in blue0):
        print("disconnect call: ",target)
        disconnect_target(target)
    #disconnect(matrix,src0)
    #print("disconnected")
#total number of white on corners
len_WN=len(white_noncorner)
print("white at green corners are in jail now")
def least_distance(white_noncorner,src):
    start=src
    l,m=start//n,start%n
    node=white_noncorner[0]
    x=1000
    for i in white_noncorner:
        if(abs(l-i//n)+abs(m-i%n)<=x):
            x=abs(l-i//n)+abs(m-i%n)
            node=i
    print("least distance white non corner is:",node)
    return node
#removed shape from block is updated->
def update_weight(node):
    for c in range(n*n):
        if(matrix[node][c]==10):
            matrix[node][c]=2
            #matrix[c][node]=2
        
def reupdate_weight(node):
    for c in range(n*n):
        if(matrix[node][c]==2):
            matrix[node][c]=10
            #matrix[c][node]=10
    
for f in track_removed_wc:
    _=revealed(f)
print("shape of corner after revealing",shape)
print("...............now target will be non green white/WEAPON and their corresponding shape/HOCRUX......................")
white_noncorner_rem=[]
for m in range(len_WN):
        src_cen,v=bot_pos()
        while(src_cen==[-1,-1] and v==-1j):
            src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row :column",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        print("source node is: ",int(src))
        #least distance white at non corner is target
        target=least_distance(white_noncorner,src)
        white_noncorner.remove(target)
        connect_node(target,matrix)
        print("target non corner white:",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target]
        while(x!=src):
            print(" x is:",x)
            path.insert(0,x)
            x=parent[x]
        #path.insert(0,src)
        print("path to follow: ",path)
        movement(path)
        print("down0")
        ser.write(b'd')
        time.sleep(1.3)
        ser.write(b's')
        ser.write(b'l')
        time.sleep(1.5)
        ser.write(b's')
        #move the box by 90
        
        shape_obtained=revealed(target)
        print("shape_obtained:",shape_obtained)
        #update the weight corresponding to this shape
        
        k=0
        for t in green:
            if(shape_obtained==shape[t//n][t%n]):
                k=k+1
                target=t
        #ser.write(b'r')
        #time.sleep(1.2) 
        if(k==0):
            print("shape not found on green nodes")
            ser.write(b'r')
            time.sleep(1.5)
            ser.write(b'u')
            time.sleep(1.1)
            ser.write(b'b')
            time.sleep(0.7)
            ser.write(b's')
            white_noncorner_rem.append(target)
            disconnect_target(target)
            continue
        for p in range(n*n):
            if(shape[p//n][p%n]==shape_obtained):
                print("nodes to update",p)
                update_weight(p)
        #shape is found and target is green cell that matches with revealed shape
        print("blink green")
        ser.write(b'G')
        time.sleep(0.5)
        ser.write(b'r')
        time.sleep(1.5)
        ser.write(b's')
        time.sleep(0.25)
        ser.write(b'f')
        time.sleep(1.5)
        ser.write(b's')
        src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row of revealed :column of revealed",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        print("source node for green corner and target is:",int(src)," ",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target] 
        while(x!=src):
            print(" x is:",x)
            path.insert(0,x)
            x=parent[x]
        #path.insert(0,src)
        print("path to follow: ",path)
        movement(path)
        print("up1")
        print("blink red led")
        ser.write(b'R')
        time.sleep(0.5)
        ser.write(b'u')
        time.sleep(1.1)
        ser.write(b's')
        ser.write(b'b')
        time.sleep(1.5)
        ser.write(b's')
        '''else:
            print("down1")
            ser.write(b'd')
            time.sleep(1.1)
            ser.write(b'l')
            time.sleep(0.5)'''
        for p in range(n*n):
            if(shape[p//n][p%n]==shape_obtained):
                reupdate_weight(p)
        #destination is achieved ,if target was in greeen or blue box disconnect that target/box
        if(target in Green_blue):
            disconnect_target(target)
            shape[target//n][target%n]=-1
        src_cen0,v0=bot_pos()
        node_column0=src_cen0[0]//width
        node_row0=src_cen0[1]//height
        print("row :column0",node_row0,", ",node_column0)
        src0=int(node_row0*n+node_column0)
        print("source node is: ",int(src0))
        #disconnect(matrix,src0)

#transfer the remaining white block at green position to jail ,disconnect the jail and then transfer non corner white at the green cell matching with shape

#NOT NECESSARY FOR FINAL PS ,BECAUSE NO JAIL WAS BLOCKED FROM ALL SIDES INITAILLY
'''if(len(white_corner)!=0):
        #go to left corner white
        print("white block at green still remaining: ",len(white_corner))
        target=white_corner[0]
        print("white non corner remaining at node(TARGET): ",target)
        src_cen,v=bot_pos()
        while(src_cen==[-1,-1] and v==-1j):
            src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row :column",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        print("source node is: ",int(src))
        #white_corner.remove(target)
        connect_node(target,matrix)
        print("target non corner white:",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target]
        while(x!=src):
            print(" x is:",x)
            path.insert(0,x)
            x=parent[x]
        #path.insert(0,src)
        print("path to follow: ",path)
        movement(path)
        print("down0")###############
        ser.write(b'd')
        time.sleep(1.1)
        ser.write(b'b')
        time.sleep(0.7)
        ser.write(b's')
        target=left_blue[0]
        #go to left blue cell
        left_blue.remove(target)
        print("jail remaining at node : ",target)
        src_cen,v=bot_pos()
        while(src_cen==[-1,-1] and v==-1j):
            src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row :column",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        print("source node is: ",int(src))
        connect_node(target,matrix)
        print("target non corner white:",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target]
        while(x!=src):
            print(" x is:",x)
            path.insert(0,x)
            x=parent[x]
        #path.insert(0,src)
        print("path to follow: ",path)
        movement(path)
        print("up")
        ser.write(b'B')
        time.sleep(0.5)
        ser.write(b'u')
        time.sleep(1.3)
        ser.write(b'b')
        time.sleep(0.7)
        ser.write(b's')
        #if(target in Green_blue):#####################
        disconnect_target(target)
        #target the left weapon
        print("targetting the left weapon")
        target=white_noncorner_rem[0]
        white_noncorner_rem.remove(target)
        src_cen,v=bot_pos()
        while(src_cen==[-1,-1] and v==-1j):
            src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row :column",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        connect_node(target,matrix)
        print("targetting corner left cell:",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target]
        while(x!=src):
            print(" x is:",x)
            path.insert(0,x)
            x=parent[x]
        #path.insert(0,src)
        print("path to follow: ",path)
        movement(path)
        print("down0")
        ##################
        ser.write(b'd')
        time.sleep(1.3)
        print("revealing weapon")#>>>>>>>>>>>>>>>>>>>>>>NOW LAST WEAPON WILL BE CARRIED TO HORCRUX
        
        ser.write(b'l')
        time.sleep(2.5)
        #move the box by 90
        ser.write(b'G')
        time.sleep(0.5)
        ser.write(b'r')
        time.sleep(2.5)
        ser.write(b'f')
        time.sleep(0.8)
        shape_obtained=revealed(target)
        print("shape_obtained:",shape_obtained)
        #update the weight corresponding to this shape 
        
        '''k=0
        for t in green:
            if(shape_obtained==shape[t//n][t%n]):
                k=k+1
                target=t
        #ser.write(b'r')
        #time.sleep(1.2) 
        if(k==0):
            print("shape not found on green nodes")
            ser.write(b'r')
            time.sleep(2.7)
            ser.write(b'u')
            time.sleep(1.1)
            ser.write(b'b')
            time.sleep(0.8)
            disconnect_target(target)'''
        target=white_corner[0]
        
        for p in range(n*n):
            if(shape[p//n][p%n]==shape_obtained):
                print("nodes to update",p)
                update_weight(p)
        connect_node(target,matrix)
        '''ser.write(b'r')
        time.sleep(2.7)
        ser.write(b'f')
        time.sleep(0.73)'''
        #go to left white corner/HOCRUX
        src_cen,v=bot_pos()
        node_column=src_cen[0]//width
        node_row=src_cen[1]//height
        print("row of revealed :column of revealed",node_row,", ",node_column)
        src=int(node_row*n+node_column)
        print("source node for green corner and target is:",int(src)," ",target)
        nodes=matrix.shape[0]
        visited=[False]*nodes
        distance=[1000]*nodes
        parent=dijkstras(matrix,src,target,nodes,visited,distance)
        #to remove target point.....
        path=[target]
        x=parent[target] 
        while(x!=src):
            print(" x is:",x)
            path.insert(0,x)
            x=parent[x]
        #path.insert(0,src)
        print("path to follow: ",path)
        movement(path)
        ser.write(b'R')
        time.sleep(0.5)

else:
    print("no corner white box is remaining")
    
        
a=np.zeros([n,n],dtype=int)
k=0
#//////
for i in range(n):
    for j in range(n):
        a[i][j]=k
        k=k+1
print(a)''''
ser.write(b'u')
time.sleep(1.1)
ser.write(b'b')
time.sleep(0.7)
ser.write(b's')
ser.write(b'E')
time.sleep(3)
ser.write(b's')

