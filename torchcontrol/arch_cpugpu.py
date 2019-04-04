# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:05:51 2019

@author: Zymieth
"""
from .imports import *
from .predictors import MLP, CNN
from scipy.integrate import solve_ivp

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# main class for all the ugly shit
class HDNN(nn.Module):
    '''
    High level wrapper for Hamiltonian Differential Neural Networks
    
    :dense_layer: list of dimensions of linear layers of the chosen HDNN predictor \
    (e.g [12,24,2] for 12-dimensional inputs and output size 2)
    :p_type: imported nn.Module class with forward method to be used as predictor
    :p_args: arguments for p_type. List format expected, unpacked with *args
    :hparams: list [a,b,c] of loss function hyperparameters
    :beta: beta of F function of the weight dynamics
    :p_module: module name from which the predictor class is imported 
    '''
    def __init__(self, p_type, p_args, hparams, beta, p_module=__name__, odeint='cpu'):
      
        # initialize superclass method
        super().__init__()
        
        # predictor initialization
        self.p_type = getattr(sys.modules[p_module], p_type)
        self.p_args = p_args
        
        self.predictor = self.p_type(*self.p_args).to(device)
        self.len = len(self.predictor.state_dict())
        self.lenp = sum(1 for _ in iter(self.predictor.parameters()))
               
        # simple assignment routine
        self.hparams = hparams
        self.dJ,self.dJddw = [], []
        self.beta = beta
           
        # creating flattened versions of self.w and self.wdot for gradient and state computations
        self.flat_w = self.flattenParamVector() 
        self.flat_wdot = torch.rand(self.flat_w.shape)
     
        self.odeint = odeint
             
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
        for i in range(self.lenp):
            param = next(itr)
            w.append(param.to(device))
            if first_instance:
                wdot.append(torch.rand((param.shape)).to(device))
        if velocity == True: return w,wdot
        else: return w
    
    def flattenParamVector(self):
        itr = iter(self.predictor.parameters())         
        w = next(itr).view(-1)
        for i in range(1,self.lenp):
            w = torch.cat((w,(next(itr).view(-1))))
        return w
    
    def flattenParamVector2(self):
        itr = iter(self.state_dict().values())         
        w = next(itr).view(-1)
        for i in range(1,self.lenp):
            w = torch.cat((w,(next(itr).view(-1))))
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
        ## BIG CHANGE HERE ##
        itr = iter(self.predictor.parameters())
        ##itr = iter(self.state_dict())##
        dJddw = 2.*self.hparams[1]*self.flat_wdot
        #for idx,params in enumerate(self.wdot):
        #dJddw.append(2.*self.hparams[1]*self.wdot[idx])
        dJ_reg = 2.*self.hparams[2]*self.flat_w
        dJ = list(map(add,[next(itr).grad for i in range(self.len)],dJ_reg))
        self.dJ,self.dJddw = dJ, dJddw
        
    def additionalTermsLoss(self):
        return torch.add(self.hparams[1]*torch.dot(self.flat_wdot,self.flat_wdot),\
        self.hparams[2]*torch.dot(self.flat_w,self.flat_w))
    
    def assignNewState(self, xi):
        self.flat_w = torch.Tensor(xi[:len(self.flat_w)])
        self.flat_wdot = torch.Tensor(xi[len(self.flat_w):2*len(self.flat_w)])
        ### CURRENTLY HANDLED WITHOUT SELF.W
        #for i in range(self.getLength()):
        #    self.w[i] = self.flat_w[k:k+torch.numel(self.w[i])].view(self.w[i].shape)
        #    self.wdot[i] = self.flat_wdot[k:k+torch.numel(self.wdot[i])].view(self.wdot[i].shape)
        #    k += torch.numel(self.w[i])
        
    def loadStateDict(self):    
        new_state_dict = self.makeStateDict()
        del self.predictor
        self.predictor = self.p_type(*self.p_args).to(device)
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
        for i in range(1,self.len):
            dJ = torch.cat((dJ,self.dJ[i].view(-1)))
            #dJddw = torch.cat((dJddw,self.dJddw[i].view(-1)))
        return dJ.to(device), self.dJddw.to(device)
    
    def assignFlatGradient(self):
        self.flat_dJ,self.flat_dJddw = self.flattenGradient()
    
    def getConcatGradient(self):
        return torch.cat((self.flat_dJ, self.flat_dJddw))
    
    def fixInputOutput(self,x,y):
        self.x = x
        self.y = y
        
    def setXi(self):
        self.xi = torch.cat((self.flat_w.to(device), self.flat_wdot.to(device)))
        
    def getParamShape(self):
        return self.shape
    
    def pred_accuracy(self,testloader):
        tot = 0
        count = 0
        for i,d in enumerate(testloader):
            x,y = d
            x,y = x.to(0),y.to(0)
            _, idx = torch.max(torch.exp(self.predictor.forward(x)), 1)  
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
        
        #update internal flat w vectors
        if self.odeint == 'gpu': self.assignNewState(xi.cpu())
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
        if self.time % 10 == 0: print("odeint iter: {} ".format(self.time))
        if self.odeint == 'cpu': dxdt = dxdt.detach().cpu().numpy()
        return dxdt
    
    def fit(self, trainloader, epoch=3, time_delta=1, iter_accuracy=10, ode_t=0.25, ode_step=10, criterion='nll'):
        '''
        :trainloader: DataLoader with training data
        :epoch: number of training epochs
        :time_delta: time steps required for a single recording of loss and parameters. Higher is better for speed. If None, no plotting
        :method: can be either 'odeint' or 'adjoint'
        :iter_accuracy: iterations until test accuracy is displayed
        :ode_t: number of odeint time steps (per batch)
        :ode_step: number of odeint time steps (per batch)
        :criterion: nll for negative log-likelihood, mse for mean-square error
        '''        
        if criterion == 'nll': self.criterion = F.nll_loss 
        else: self.criterion = F.mse_loss         
        if self.odeint == 'cpu': t = np.linspace(0., ode_t, ode_step) 
        elif self.odeint == 'gpu': t = torch.linspace(0., ode_t, ode_step)
        if time_delta: self.time_delta = time_delta
        else: self.time_delta = float('inf')
        self.setXi()  
        
        for e in range(epoch): 
            for i, data in enumerate(trainloader):               
                x,y = data
                x,y = x.to(device),y.to(device)                
                self.fixInputOutput(x,y)              
                func = self
                if self.odeint == 'cpu': 
                    xi = solve_ivp(func, t, self.xi.cpu().detach().numpy())
                    xi = [el[-1] for el in xi.y]
                elif self.odeint == 'gpu':
                    xi = odeint(func, self.xi, t)    
                self.xi = torch.Tensor(xi)
                self.assignNewState(xi)
                del xi
                self.loadStateDict()
                self.count += 1                
                if self.count % iter_accuracy == 0 and self.count != 0:
                    print('Number of odeint and parameters reassignment iterations: {}'.format(self.count))
                    print('In-training accuracy estimate: {}'.format(self.pred_accuracy(trainloader)))

    def observe(self, x0, labels, epoch=3, dynamics_delta_t = 0.1, ode_t = 10, ode_step = 100, loss_record_interval=1):
              
        self.criterion = F.mse_loss 
        self.setXi()
        
        if loss_record_interval: self.time_delta = loss_record_interval
        else: self.time_delta = float('inf')
        
        if self.odeint == 'cpu': t = np.linspace(0., ode_t, ode_step) 
        elif self.odeint == 'gpu': t = torch.linspace(0., ode_t, ode_step)
        x = x0
        trajectory = [x0]
              
        for e in range(epoch):
            for i, y in enumerate(labels):
                self.fixInputOutput(x,y) 
                func = self
                if self.odeint == 'cpu': 
                    xi = solve_ivp(func, t, self.xi.cpu().detach().numpy())
                    xi = [el[-1] for el in xi.y]
                elif self.odeint == 'gpu':
                    xi = odeint(func, self.xi, t)                 
                self.xi = torch.Tensor(xi)
                self.assignNewState(xi) 
                self.loadStateDict()
                with torch.no_grad():
                    x = self.predictor.forward(x)
                    trajectory.append(x)                    
                self.predictor.zero_grad()
                self.count += 1  
                try:              
                    if self.count % iter_accuracy == 0 and self.count != 0:
                        print('Number of odeint and parameters reassignment iterations: {}'.format(self.count))
                        print('In-training accuracy estimate: {}'.format(self.pred_accuracy(trainloader)))
                except:
                    pass
        return trajectory
                    

# to be deleted    
class HDNN_Observer(HDNN):
    def __init__(self, p_type, p_args, hparams, beta, odeint='cpu'):
        super(HDNN_Observer, self).__init__(p_type, p_args, hparams, beta, odeint)   

    def observe(self, x0, labels, epoch=3, dynamics_delta_t = 0.1, ode_t = 10, ode_step = 100, loss_record_interval=1):
              
        self.criterion = F.mse_loss 
        self.setXi()
        
        if loss_record_interval: self.time_delta = loss_record_interval
        else: self.time_delta = float('inf')
        
        if self.odeint == 'cpu': t = np.linspace(0., ode_t, ode_step) 
        elif self.odeint == 'gpu': t = torch.linspace(0., ode_t, ode_step)
        x = x0
        trajectory = [x0]
              
        for e in range(epoch):
            for i, y in enumerate(labels):
                self.fixInputOutput(x,y) 
                func = self
                if self.odeint == 'cpu': 
                    xi = solve_ivp(func, t, self.xi.cpu().detach().numpy())
                    xi = [el[-1] for el in xi.y]
                elif self.odeint == 'gpu':
                    xi = odeint(func, self.xi, t)                 
                self.xi = torch.Tensor(xi)
                self.assignNewState(xi) 
                self.loadStateDict()
                with torch.no_grad():
                    x = self.predictor.forward(x)
                    trajectory.append(x)                    
                self.predictor.zero_grad()
                self.count += 1  
                try:              
                    if self.count % iter_accuracy == 0 and self.count != 0:
                        print('Number of odeint and parameters reassignment iterations: {}'.format(self.count))
                        print('In-training accuracy estimate: {}'.format(self.pred_accuracy(trainloader)))
                except:
                    pass
        return trajectory

class HDNN_LSTM_Observer(HDNN_Observer):
    def __init__(self, seq_length):
        super().__init__()
    
    def observe(self, x0, labels, epoch=3, dynamics_delta_t=0.1, ode_t=10, ode_step=100, loss_record_interval=1):                      
        self.criterion = F.mse_loss 
        self.setXi()
        
        if loss_record_interval: self.time_delta = loss_record_interval
        else: self.time_delta = float('inf')
        
        if self.odeint == 'cpu': t = np.linspace(0., ode_t, ode_step) 
        elif self.odeint == 'gpu': t = torch.linspace(0., ode_t, ode_step)
        x = x0
        trajectory = [x0]
              
        for e in range(epoch):
            for i, y in enumerate(labels):
                self.fixInputOutput(x,y) 
                func = self
                if self.odeint == 'cpu': 
                    xi = solve_ivp(func, t, self.xi.cpu().detach().numpy())
                    xi = [el[-1] for el in xi.y]
                elif self.odeint == 'gpu':
                    xi = odeint(func, self.xi, t)                 
                self.xi = torch.Tensor(xi)
                self.assignNewState(xi) 
                self.loadStateDict()
                with torch.no_grad():
                    x = self.predictor.forward(x)
                    trajectory.append(x)                    
                self.predictor.zero_grad()
                self.count += 1  
                try:              
                    if self.count % iter_accuracy == 0 and self.count != 0:
                        print('Number of odeint and parameters reassignment iterations: {}'.format(self.count))
                        print('In-training accuracy estimate: {}'.format(self.pred_accuracy(trainloader)))
                except:
                    pass
        return trajectory    
            
            