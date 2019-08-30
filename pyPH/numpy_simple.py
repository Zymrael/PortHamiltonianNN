"""Low level implementation of a linear neural network optimized with Port-Hamiltonian
dynamics."""

import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm_notebook as tqdm


def simple_net(u, w):
    """Simple linear neural network"""
    # inputs
    u1, u2 = u[0], u[1]
    # weights of y1
    w11, w12, w13 = w[0], w[1], w[2]
    # weights of y2
    w21, w22, w23 = w[3], w[4], w[5]

    y1 = w11*u1 + w12*u2 + w13
    y2 = w21*u1 + w22*u2 + w23
    return np.array([y1,y2]).T


def loss(x, u, yh, a, b, c, n):
    """Computes loss J"""
    w, dw = x[0:n], x[n:2*n]
    J = np.array(yh - simple_net(u, w))
    loss = a*J.dot(J) + b*dw.dot(dw) + c*w.dot(w)
    return loss


def gradient(x, u, yh, a, b, c):
    """Computes gradient of the loss J"""
    # x contains weights AND momenta
    n = len(x)//2
    # weights and their velocities
    w, dw = x[0:n], x[n:2*n]
    # inputs
    u1, u2 = u[0], u[1]
    # ground truth (reference outputs)
    yh1, yh2 = yh[0], yh[1]
    # weights of y1
    w11, w12, w13 = w[0], w[1], w[2]
    # weights of y2
    w21, w22, w23 = w[3], w[4], w[5]

    y = simple_net(u, w)
    # gradient computation
    dJ_w = 2.*a*np.array([u1*(y[0]-yh1), u2*(y[0]-yh1), y[0]-yh1, u1*(y[1]-yh2), u2*(y[1]-yh2), y[1]-yh2]).T
    #
    dJ_dw = np.array([2.*b*dw[0], 2.*b*dw[1], 2.*b*dw[2], 2.*b*dw[3], 2.*b*dw[4], 2.*b*dw[5]]).T
    dJ = np.hstack((dJ_w, dJ_dw)).T
    # regularisation term
    dJr = np.array([2.*c*w11, 2.*c*w12, 2.*c*w13, 2.*c*w21, 2.*c*w22, 2.*c*w23, 0., 0., 0., 0., 0., 0.])
    return dJ + dJr


def hamiltonian_model(x, u, yh, beta, a, b, c):
    """Defines ODE of the PH Model"""
    n = len(x)//2
    # Compute the gradient
    dJ = gradient(x, u, yh, a, b, c)
    # Compute derivative
    dwdt = dJ[n:2*n]/b
    ddwdt = -dJ[0:n] - beta*dJ[n:2*n]/b
    dxdt = np.hstack((dwdt, ddwdt))
    return dxdt


def loss_batch(bs,x,U,Yh,a,b,c,n):
    """Calculates loss J for a batch"""
    J_tot = 0
    for i in range(bs):
        u = U[i]
        yh = Yh[i]
        J_tot = J_tot + loss(x,u,yh,a,b,c,n)
    return J_tot/bs


def gradient_batch(bs,x,U,Yh,a,b,c,n):
    dJ_tot = np.zeros((1,2*n))
    dJ_tot = dJ_tot[0] 
    for i in range(bs):
        u = U[i]
        yh = Yh[i]
        dJ_tot = dJ_tot + gradient(x,u,yh,a,b,c,n)

    return dJ_tot/bs 


def ham_mod_batch(x,t,bs,U,Yh,beta,a,b,c,n):
    # define the matrix F
    On = np.zeros((n,n))
    In = np.eye(n)
    B = np.array(beta*np.eye(n))
    F = np.vstack((np.hstack((On,In)),np.hstack((-In,-B))))
    # Compute the gradient
    dJ = gradient_batch(bs,x,U,Yh,a,b,c,n)
    # Compute derivative
    dxdt = F.dot(dJ)
    return dxdt



def train(X,y,bs,epochs,x0,a,b,c,beta,n,t):
    """Trains a PHNN"""
    N_tot = len(X)
    N_batch = int(N_tot/bs)
    Ub = X.reshape(N_batch,bs,2)
    Yhb = y.reshape(N_batch,bs,2)
    # Initialize variable in which to store results
    xf = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],x0], dtype='float')
    tf = np.array([0.])
    J = np.array([0.])
    xep = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],x0], dtype='float')
    Jep = loss_batch(bs,np.array(x0),Ub[0],Yhb[0],a,b,c,n)
    for j in tqdm(range(epochs)):
        for i in tqdm(range(N_batch)):
            U = Ub[i]
            Yh = Yhb[i]
            x0 = xf[-1] 
            sol = odeint(ham_mod_batch, x0, t, args=(bs,U,Yh,beta,a,b,c,n))
            xf = np.vstack((xf,sol))
            tf = np.hstack((tf,t+tf[-1]))
            N = len(sol)
            Ji = np.zeros((N,1))
            for i in range(N):
                Ji[i] = loss_batch(bs,sol[i],U,Yh,a,b,c,n)
            J = np.vstack((J,Ji))
        xep = np.vstack((xep,xf[-1]))
        Jep = np.vstack((Jep,J[-1]))
    xf = xf[2:-1]
    J = J[1:-1]
    tf = tf[1:-1]
    xep = xep[1:-1]
    return tf, xf, J, xep, Jep


def test(x,Xh,yh,trained,train_test):
    """Performs evaluation on out-of-sample data"""
    N = len(yh)
    count_predicted = 0
    for i in range(N):
        y = simple_net(Xh[i],x)
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
def out_gradient(x,u,yh,a,b,c,n,C):
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
    xh = simple_net(u,w)
    y = xh[0]*C[0] + xh[1]*C[1]

    dJ_w = 2.*a*np.array([u1*(y[0]-yh1),u2*(y[0]-yh1),y[0]-yh1,u1*(y[1]-yh2),u2*(y[1]-yh2),y[1]-yh2]).T
    #
    dJ_dw = np.array([2.*b*dw[0],2.*b*dw[1],2.*b*dw[2],2.*b*dw[3],2.*b*dw[4],2.*b*dw[5]]).T
    dJ = np.hstack((dJ_w,dJ_dw)).T
    # regularisation term
    dJr = np.array([2.*c*w11,2.*c*w12,2.*c*w13,2.*c*w21,2.*c*w22,2.*c*w23,0.,0.,0.,0.,0.,0.])
    return dJ+dJr


def out_gradient_batch(bs,x,U,Yh,a,b,c,n):
    dJ_tot = np.zeros((1,2*n))
    dJ_tot = dJ_tot[0] 
    for i in range(bs):
        u = U[i]
        yh = Yh[i]
        dJ_tot = dJ_tot + out_gradient(x,u,yh,a,b,c,n)
    return dJ_tot/bs 


def out_ham_mod_batch(x,t,bs,U,Yh,beta,a,b,c,n):
    # define the matrix F
    On = np.zeros((n,n))
    In = np.eye(n)
    B = np.array(beta*np.eye(n))
    F = np.vstack((np.hstack((On,In)),np.hstack((-In,-B))))
    # Compute the gradient
    dJ = out_ham_mod_batch(bs,x,U,Yh,a,b,c,n)
    # Compute derivative
    dxdt = F.dot(dJ)
    return dxdt
