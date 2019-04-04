import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm_notebook as tqdm


############################################################################
###################### BASIC FUNCTIONS #####################################
# Define linear neural network 
def DumbNet(u,w):
    # inputs
    u1 = u[0]
    u2 = u[1]
    # weights of y1
    w11 = w[0]
    w12 = w[1]
    w13 = w[2]
    # weights of y2
    w21 = w[3]
    w22 = w[4]
    w23 = w[5]
    ############
    # Compute output
    y1 = w11*u1 + w12*u2 + w13
    y2 = w21*u1 + w22*u2 + w23
    return np.array([y1,y2]).T

# Compute the value of the loss J
def Loss(x,u,yh,a,b,c,n):
    w = x[0:n]
    dw = x[n:2*n]
    J1 = np.array(yh-DumbNet(u,w))
    #print(J1)
    J = a*J1.dot(J1) + b*dw.dot(dw) + c*w.dot(w)
    return J

# Compute the gradient of the loss J
def Gradient(x,u,yh,a,b,c,n):
    # weights and their velocities
    w = x[0:n]
    dw = x[n:2*n]
    # inputs
    u1 = u[0]
    u2 = u[1]
    # ground truth (reference outputs)
    yh1 = yh[0]
    yh2 = yh[1]
    # weights of y1
    w11 = w[0]
    w12 = w[1]
    w13 = w[2]
    # weights of y2
    w21 = w[3]
    w22 = w[4]
    w23 = w[5]
    #############################################
    # Compute gradient of J
    y = DumbNet(u,w)
    dJ_w = 2.*a*np.array([u1*(y[0]-yh1),u2*(y[0]-yh1),y[0]-yh1,u1*(y[1]-yh2),u2*(y[1]-yh2),y[1]-yh2]).T
    #
    dJ_dw = np.array([2.*b*dw[0],2.*b*dw[1],2.*b*dw[2],2.*b*dw[3],2.*b*dw[4],2.*b*dw[5]]).T
    dJ = np.hstack((dJ_w,dJ_dw)).T
    # regularisation term
    dJr = np.array([2.*c*w11,2.*c*w12,2.*c*w13,2.*c*w21,2.*c*w22,2.*c*w23,0.,0.,0.,0.,0.,0.])
    #print(dJ,dJr)
    return dJ+dJr

# Define the PH model (ODE)
def HamiltonianModel(x,t,u,yh,beta,a,b,c,n):
    # define the matrix F
    On = np.zeros((n,n))
    In = np.eye(n)
    B = np.array(beta*np.eye(n))
    #F = np.vstack((np.hstack((On,In/b)),np.hstack((-In,-B))))
    # Compute the gradient
    dJ = Gradient(x,u,yh,a,b,c,n)
    # Compute derivative
    #dxdt = F.dot(dJ)
    dwdt = dJ[n:2*n]/b
    ddwdt = -dJ[0:n] - beta*dJ[n:2*n]/b
    dxdt = np.hstack((dwdt,ddwdt))
    return dxdt

############################################################################
###################### BATCH TRAINING FUNCTIONS ############################
def LossBatch(bs,x,U,Yh,a,b,c,n):
    J_tot = 0
    for i in range(bs):
        #print(i)
        u = U[i]
        yh = Yh[i]
        #print(u,yh)
        J_tot = J_tot + Loss(x,u,yh,a,b,c,n)
    # print(dJ_tot[0]/bs)
    return J_tot/bs

def GradientBatch(bs,x,U,Yh,a,b,c,n):
    dJ_tot = np.zeros((1,2*n))
    dJ_tot = dJ_tot[0] 
    for i in range(bs):
        #print(i)
        u = U[i]
        yh = Yh[i]
        #print(u,yh)
        dJ_tot = dJ_tot + Gradient(x,u,yh,a,b,c,n)
        #print(Gradient(x,u,yh,a,b,c,n))
    return dJ_tot/bs 

def HamModBatch(x,t,bs,U,Yh,beta,a,b,c,n):
    # define the matrix F
    On = np.zeros((n,n))
    In = np.eye(n)
    B = np.array(beta*np.eye(n))
    F = np.vstack((np.hstack((On,In)),np.hstack((-In,-B))))
    # Compute the gradient
    dJ = GradientBatch(bs,x,U,Yh,a,b,c,n)
    #print(dJ)
    # Compute derivative
    dxdt = F.dot(dJ)
    return dxdt

    ###############################################################
    ############# TRAINING ########################################

def train(X,y,bs,epochs,x0,a,b,c,beta,n,t):
    N_tot = len(X)
    N_batch = int(N_tot/bs)
    Ub = X.reshape(N_batch,bs,2)
    Yhb = y.reshape(N_batch,bs,2)
    # Initialise variable where to store results
    xf = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],x0], dtype='float')
    tf = np.array([0.])
    J = np.array([0.])
    xep = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],x0], dtype='float')
    Jep = LossBatch(bs,np.array(x0),Ub[0],Yhb[0],a,b,c,n)
    for j in tqdm(range(epochs)):
        for i in tqdm(range(N_batch)):
            U = Ub[i]
            Yh = Yhb[i]
            x0 = xf[-1] 
            sol = odeint(HamModBatch, x0, t, args=(bs,U,Yh,beta,a,b,c,n))
            xf = np.vstack((xf,sol))
            tf = np.hstack((tf,t+tf[-1]))
            N = len(sol)
            Ji = np.zeros((N,1))
            for i in range(N):
                Ji[i] = LossBatch(bs,sol[i],U,Yh,a,b,c,n)
            J = np.vstack((J,Ji))
        xep = np.vstack((xep,xf[-1]))
        Jep = np.vstack((Jep,J[-1]))
    xf = xf[2:-1]
    J = J[1:-1]
    tf = tf[1:-1]
    xep = xep[1:-1]
    return tf, xf, J, xep, Jep


def test(x,Xh,yh,trained,train_test):
    N = len(yh)
    count_predicted = 0
    for i in range(N):
        y = DumbNet(Xh[i],x)
        if (y[0]>y[1]) and (yh[i,0]>yh[i,1]):
            count_predicted += 1
    accuracy = count_predicted*100./N
    #
    if train_test:
        if trained:
            print('Post-training train set accuracy:',accuracy,'%')
        else:
            print('Pre-training train set accuracy:',accuracy,'%')
    else:
        if trained:
            print('Post-training test accuracy:',accuracy,'%')
        else:
            print('Pre-training test accuracy:',accuracy,'%')
    return accuracy

    ################### EXPERIMENTAL ######################

def OutGradient(x,u,yh,a,b,c,n,C):
    # weights and their velocities
    w = x[0:n]
    dw = x[n:2*n]
    # inputs
    u1 = u[0]
    u2 = u[1]
    # ground truth (reference outputs)
    yh1 = yh[0]
    yh2 = yh[1]
    # weights of y1
    w11 = w[0]
    w12 = w[1]
    w13 = w[2]
    # weights of y2
    w21 = w[3]
    w22 = w[4]
    w23 = w[5]
    #############################################
    # Compute gradient of J
    xh = DumbNet(u,w)
    y = xh[0]*C[0] + xh[1]*C[1]

    dJ_w = 2.*a*np.array([u1*(y[0]-yh1),u2*(y[0]-yh1),y[0]-yh1,u1*(y[1]-yh2),u2*(y[1]-yh2),y[1]-yh2]).T
    #
    dJ_dw = np.array([2.*b*dw[0],2.*b*dw[1],2.*b*dw[2],2.*b*dw[3],2.*b*dw[4],2.*b*dw[5]]).T
    dJ = np.hstack((dJ_w,dJ_dw)).T
    # regularisation term
    dJr = np.array([2.*c*w11,2.*c*w12,2.*c*w13,2.*c*w21,2.*c*w22,2.*c*w23,0.,0.,0.,0.,0.,0.])
    #print(dJ,dJr)
    return dJ+dJr

def OutGradientBatch(bs,x,U,Yh,a,b,c,n):
    dJ_tot = np.zeros((1,2*n))
    dJ_tot = dJ_tot[0] 
    for i in range(bs):
        #print(i)
        u = U[i]
        yh = Yh[i]
        #print(u,yh)
        dJ_tot = dJ_tot + OutGradient(x,u,yh,a,b,c,n)
        #print(Gradient(x,u,yh,a,b,c,n))
    return dJ_tot/bs 

def OutHamModBatch(x,t,bs,U,Yh,beta,a,b,c,n):
    # define the matrix F
    On = np.zeros((n,n))
    In = np.eye(n)
    B = np.array(beta*np.eye(n))
    F = np.vstack((np.hstack((On,In)),np.hstack((-In,-B))))
    # Compute the gradient
    dJ = OutGradientBatch(bs,x,U,Yh,a,b,c,n)
    #print(dJ)
    # Compute derivative
    dxdt = F.dot(dJ)
    return dxdt

def OutTrain(X,y,bs,epochs,x0,a,b,c,beta,n,t):
    N_tot = len(X)
    N_batch = int(N_tot/bs)
    Ub = X.reshape(N_batch,bs,2)
    Yhb = y.reshape(N_batch,bs,2)
    # Initialise variable where to store results
    xf = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],x0], dtype='float')
    tf = np.array([0.])
    J = np.array([0.])
    for j in tqdm(range(epochs)):
        for i in tqdm(range(N_batch)):
            U = Ub[i]
            Yh = Yhb[i]
            x0 = xf[-1] 
            sol = odeint(OutHamModBatch, x0, t, args=(bs,U,Yh,beta,a,b,c,n))
            xf = np.vstack((xf,sol))
            tf = np.hstack((tf,t+tf[-1]))
            N = len(sol)
            Ji = np.zeros((N,1))
            for i in range(N):
                Ji[i] = OutLossBatch(bs,sol[i],U,Yh,a,b,c,n)
            J = np.vstack((J,Ji))
    xf = xf[2:]
    J = J[1:]
    tf = tf[1:]
    return tf, xf, J
