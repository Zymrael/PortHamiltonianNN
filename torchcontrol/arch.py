# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 09:13:37 2019

@author: Zymieth
"""
from. imports import *
from .predictors import MLP, CNN

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# main class for all the ugly shit
class HDNN(nn.Module):
    '''
    High level wrapper of a Hamiltonian Differential Neural Network
    
    :dense_layer: list of dimensions of linear layers of the chosen HDNN predictor \
    (e.g [12,24,2] for 12-dimensional inputs and output size 2)
    :predictor_type: 'MLP' for dense network, 'CNN' for convolutional network
    :hparams: list [a,b,c] of loss function hyperparameters
    :beta: beta of F function of the weight dynamics
    
    NOTE: c term of the loss NOT implemented yet (no regularization). CUDA memory requirement for big 
    nets is rather big (not optimized)
    '''
    def __init__(self, layers, predictor_type, hparams, beta , CNN_dense = None, softmax=True):
      
        # initialize superclass method
        super().__init__()
        
        # create an attribute for HDNN for its predictor
        self.layers = layers
        self.predictor_type = predictor_type
        self.softmax = softmax
        
        if self.predictor_type == 'MLP': self.predictor = MLP(self.layers,self.softmax).to(device)
        else: self.predictor = CNN(self.layers,CNN_dense).to(device)
        
        # attribute self.len: each hidden layer has its own weight and bias tensors in self.predictor.state_dict() \
        # thus 2*(number of layers - 1)
        self.len = 2*(len(layers)-1)
        
        # atm useless routine to calculate shapes of predictor.parameters() tensors at different layers
        # UNUSED
        #shape = []
        #itr = iter(self.predictor.parameters())
        #for i in range(self.len):
        #    p = next(itr)
        #    shape.append(p.shape)
        #self.shape = shape
        
        # simple assignment routine
        self.hparams = hparams
        self.dJ,self.dJddw = [], []
        self.beta = beta
           
        # creating flattened versions of self.w and self.wdot for gradient and state computations
        self.flat_w = self.flattenParamVector(False) 
        self.flat_wdot = torch.rand(self.flat_w.shape).to(device)
        
        #self.flat_wdot = self.flattenParamVector(True)
        #del self.w,self.wdot
        
        ### CURRENTLY UNUSED: dealing with the dynamics without F matrix
        #self.F = self.makeFMatrix() 
        
        self.count = 0
        
        #time counter for loss and parameter plotting
        self.time = 0 
        self.pLoss = []
        self.pW = []
        self.pWdot = []
        self.initializeRecord()
        
    def createStateVector(self,velocity=True,first_instance=True):
        w = []
        wdot = []
        itr = iter(self.predictor.parameters())
        for i in range(self.getLength()):
            param = next(itr)
            w.append(param.to(device))
            if first_instance:
                wdot.append(torch.rand((param.shape)).to(device))
        if velocity == True: return w,wdot
        else: return w
    
    def flattenParamVector(self,velocity=True):
        itr = iter(self.predictor.parameters())         
        if not velocity:
            w = next(itr).view(-1).to(device)
            for i in range(1,self.getLength()):
                w = torch.cat((w,(next(itr).view(-1)).to(device)))
        else:
            w = self.wdot[0].view(-1).to(device)
            for i in range(1,self.getLength()):
                w = torch.cat((w,self.wdot[i].view(-1).to(device)))
        return w

    def makeFMatrix(self):
            '''
            Subroutine to create F matrix for parameter state dynamics
            For memory efficiency, F is stored as a torch.sparse.Tensor
            '''           
            n = len(self.flat_wdot)
            #indexes of eye(n,n)
            i1 = [[0,n]]
            for i in range(1,n):
                i1.append([i,n+i])
            i1= torch.LongTensor(i1)
            # indexes of -eye(n,n)
            i2 = [[n,0]]
            for i in range(1,n):
                i2.append([n+i,i])
            i2= torch.LongTensor(i2)         
            # indexes of -beta(n,n)
            i3 = [[n,n]]
            for i in range(1,n):
                i3.append([n+i,n+i])
            i3 = torch.LongTensor(i3) 
             
            i = torch.cat((i1,i2,i3))
            
            #value list
            v = torch.Tensor(np.concatenate((np.ones(n),-1*np.ones(n),-self.beta*np.ones(n))))
            
            F = torch.sparse.FloatTensor(i.t(),v,torch.Size([2*n,2*n]))
            del i1,i2,i3,i,v
            #In = torch.eye(n).to(device)
            #On = torch.zeros((n,n)).to(device)
            #B = self.beta*In          
            #F = torch.cat((torch.cat((On,In),1),torch.cat((-In,-B),1)),0)
            return F
        
    
    def Gradient(self):
        itr = iter(self.predictor.parameters())
        dJddw = 2.*self.hparams[1]*self.flat_wdot
        #for idx,params in enumerate(self.wdot):
        #dJddw.append(2.*self.hparams[1]*self.wdot[idx])
        dJ_reg = 2.*self.hparams[2]*self.flat_w
        dJ = list(map(add,[next(itr).grad for i in range(self.getLength())],dJ_reg))
        self.dJ,self.dJddw = dJ, dJddw
        
    def additionalTermsLoss(self):
        return torch.add(self.hparams[1]*torch.dot(self.flat_wdot,self.flat_wdot),\
        self.hparams[2]*torch.dot(self.flat_w,self.flat_w))
    
    def assignNewState(self,xi):
        self.flat_w = xi[:len(self.flat_w)]
        self.flat_wdot = xi[len(self.flat_w):2*len(self.flat_w)] 
        
        ### CURRENTLY HANDLED WITHOUT SELF.W
        #for i in range(self.getLength()):
        #    self.w[i] = self.flat_w[k:k+torch.numel(self.w[i])].view(self.w[i].shape)
        #    self.wdot[i] = self.flat_wdot[k:k+torch.numel(self.wdot[i])].view(self.wdot[i].shape)
        #    k += torch.numel(self.w[i])
        
    def loadStateDict(self):    
        new_state_dict = self.makeStateDict()
        del self.predictor
        if self.predictor_type == 'MLP': self.predictor = MLP(self.layers,self.softmax).to(device)
        else: self.predictor = CNN(self.layers,CNN_dense).to(device)
        self.predictor.load_state_dict(new_state_dict)
        del new_state_dict
        
    def makeStateDict(self):
        d = {}
        k = 0
        for i,key in enumerate(self.predictor.state_dict().keys()):       
                num_el = torch.numel(self.predictor.state_dict()[key])
                d[key] = self.flat_w[k:k+num_el].view(self.predictor.state_dict()[key].shape)
                k += num_el
        return d
    
    @staticmethod
    def makeGenerator(lt):
        for i,d in enumerate(lt):
            yield d
    
    def flattenGradient(self):
        dJ = self.dJ[0].view(-1)
        #dJddw = self.dJddw[0].view(-1)
        for i in range(1,self.getLength()):
            dJ = torch.cat((dJ,self.dJ[i].view(-1)))
            #dJddw = torch.cat((dJddw,self.dJddw[i].view(-1)))
        return dJ, self.dJddw
    
    def assignFlatGradient(self):
        self.flat_dJ,self.flat_dJddw = self.flattenGradient()
    
    def getConcatGradient(self):
        return torch.cat((self.flat_dJ,self.flat_dJddw))
    
    def fixInputOutput(self,x,y):
        self.x = x
        self.y = y
        
    def setXi(self):
        self.xi = torch.cat((self.flat_w,self.flat_wdot))

    def getLength(self):
        return self.len
    
    def getParamShape(self):
        return self.shape
    
    def pred_accuracy(self,testloader):
        tot = 0
        count = 0
        for i,d in enumerate(testloader):
            x,y = d
            x,y = x.to(0),y.to(0)
            _, idx = torch.max(torch.exp(self.predictor.forward(x)),1)  
            for i in range(len(idx)):
                if idx[i] == y[i]:
                    count += 1
                tot += 1
        return count/tot
    
    def initializeRecord(self):
        for i in range(len(self.flat_w)):
            self.pW.append([])
            self.pWdot.append([])
            
    def recordLoss(self,main_loss,additional_terms_loss,delta_t):
        if self.time % delta_t == 0:
            self.pLoss.append(main_loss+additional_terms_loss)
    
    def plotLoss(self):
        plt.plot(self.pLoss, color='red')
        plt.ylabel('Loss')
        plt.xlabel('time (delta_t units)')
        
    def recordParameters(self,pW,pWdot,delta_t):
        if self.time % delta_t == 0:
            for i in range(len(pW)):
                self.pW[i].append(pW[i].cpu().detach().numpy()) 
                self.pWdot[i].append(pWdot[i].cpu().detach().numpy()) 
        
    def plotParameters(self):
        for i in range(len(self.pW)):
            plt.plot(self.pW[i])
            
    def plotVelocities(self):
        for i in range(len(self.pW)):
            plt.plot(self.pWdot[i])
    
    def perturb(self):
        pass
    
    def forward(self,t,xi):
        #new state input
        #xi=torch.autograd.Variable(xi,requires_grad=True)
        
        #update internal flat w vectors
        self.assignNewState(xi)
        
        #use updated flat w to create new updated predictor
        self.loadStateDict()
        
        #create new optimiz for predictor and initialize
        #self.optimizer = torch.optim.Adam(self.predictor.parameters())
        #self.optimizer.zero_grad()
        
        #loss and gradient
        yhat = self.predictor.forward(self.x)
        loss = self.criterion(yhat,self.y)#self.additionalTermsLoss())
        loss.backward()
        #optional saving for plots   
        self.recordLoss(loss,self.additionalTermsLoss(),self.time_delta)
        self.recordParameters(self.flat_w,self.flat_wdot,self.time_delta)
        del loss, yhat
        
        #manual gradient w.r.t wdot and assignment to flat gradient vectors
        self.Gradient()
        self.assignFlatGradient()
        grad_flat = self.getConcatGradient()
        
        #actual ODE calcs
        dxdt = torch.zeros(len(grad_flat)).to(device)
        n = len(dxdt)//2
        dxdt[:n] = grad_flat[n:2*n]
        dxdt[n:2*n] = -1*grad_flat[:n] -1*self.beta*grad_flat[n:2*n]

        del grad_flat
        #timer for plotting purposes
        self.time += 1
        return dxdt 
    
    def fit(self, trainloader, epoch = 3, time_delta = 1, method = 'odeint', iter_accuracy = 10, ode_t = 0.25, ode_step = 10):
        '''
        :trainloader: DataLoader with training data
        :epoch: number of training epochs
        :time_delta: time steps required for a single recording of loss and parameters. Higher is better for speed. If None, no plotting
        :method: can be either 'odeint' or 'adjoint'
        :iter_accuracy: iterations until test accuracy is displayed
        :ode_t: number of odeint time steps (per batch)
        :ode_step: number of odeint time steps (per batch)
        '''
        
        self.criterion = F.nll_loss    
        t = torch.linspace(0., ode_t, ode_step)
        if time_delta: self.time_delta = time_delta
        else: self.time_delta = float('inf')
        self.setXi()
        
        for e in range(epoch): 
            for i, data in enumerate(trainloader):
                
                x,y = data
                x,y = x.to(device),y.to(device)  
                
                self.fixInputOutput(x,y)
                
                func = self
                if method == 'odeint': xi = odeint(func, self.xi, t)    
                elif method == 'adjoint': xi = odeint_adjoint(func, self.xi, t)    
                self.xi = xi[-1]
                self.assignNewState(xi[-1])
                del xi
                self.loadStateDict()
                self.count += 1                
                if self.count % iter_accuracy == 0 and self.count != 0:
                    print('Number of odeint and parameters reassignment iterations: {}'.format(self.count))
                    print('In-training accuracy estimate: {}'.format(self.pred_accuracy(trainloader)))
  

 


    

            
# class that handles the internal odeint loop
# CURRENTLY UNUSED
                    
class HamiltonianDynamics(nn.Module):
    def __init__(self, HDNN, criterion, optimizer, x, y, time_delta):
        super().__init__()
        self.parent = HDNN
        self.criterion = criterion
        self.optimizer = optimizer
        self.x = x
        self.y = y
        
        # time delta for recording purposes (loss and parameters)
        self.time_delta = time_delta
        
    def forward(self,t,xi):
        self.parent.assignNewState(xi)
        self.parent.loadStateDict()
        self.optimizer = torch.optim.Adam(self.parent.predictor.parameters())
        self.optimizer.zero_grad()
        yhat = self.parent.predictor.forward(self.x)
        loss = self.criterion(yhat,self.y)#self.additionalTermsLoss())
        loss.backward(retain_graph=True)
        
        self.parent.recordLoss(loss,self.parent.additionalTermsLoss(),self.time_delta)
        self.parent.recordParameters(self.parent.flat_w,self.parent.flat_wdot,self.time_delta)
        
        self.parent.Gradient()
        self.parent.assignFlatGradient()
        F = self.parent.makeFMatrix() 
        grad_flat = self.parent.getConcatGradient()
        dxdt = torch.matmul(F,grad_flat)
        
        #timer for parent
        self.parent.time += 1
        return dxdt   



# USELESS for now
class HDNN_adjoint(nn.Module):
    def __init__(self, grad_flat, dJ, flat_dJ,beta):
        super().__init__()
        self.grad_flat = grad_flat
        self.dJ = dJ
        self.flat_dJ = flat_dJ
        self.beta = beta
            
    def HamiltonianModel(self,t):
        #for i,d in enumerate(self.dJ):
        n = len(self.flat_dJ)
        In = torch.eye(n).to(device)
        On = torch.zeros((n,n)).to(device)
        B = self.beta*In
        F = torch.cat((torch.cat((On,In),1),torch.cat((-In,-B),1)),0)
        print(F.shape,self.grad_flat.shape)
        dxdt = torch.matmul(F,self.grad_flat)
        return dxdt                    