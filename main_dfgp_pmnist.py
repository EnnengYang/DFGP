import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
from collections import OrderedDict
import numpy as np
import argparse
from copy import deepcopy
import time

from flatness_minima import SAM
from torch.autograd import Variable

## Define MLP model
class MLPNet(nn.Module):
    def __init__(self, n_hidden=100, n_outputs=10):
        super(MLPNet, self).__init__()
        self.act=OrderedDict()
        self.lin1 = nn.Linear(784,n_hidden,bias=False)
        self.lin2 = nn.Linear(n_hidden,n_hidden, bias=False)
        self.fc1  = nn.Linear(n_hidden, n_outputs, bias=False)
    def forward(self, x):
        self.act['Lin1']=x
        x = self.lin1(x)        
        x = F.relu(x)
        self.act['Lin2']=x
        x = self.lin2(x)        
        x = F.relu(x)
        self.act['fc1']=x
        x = self.fc1(x)
        return x 
def get_model(model):
    return deepcopy(model.state_dict())
def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()
class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]

        x_ori_perm = x_ori[index, :]

        lam = lam.view((N, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b
def train (args, model, device, x, y, optimizer, criterion, inner_steps=2):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1, 28*28)
        raw_data, raw_target = data.to(device), y[b].to(device)

        # Data Perturbation Step
        # initialize lamb mix:
        N = data.shape[0]
        lam = (beta_distributions(size=N, alpha=args.mixup_alpha)).astype(np.float32)
        lam_adv = Variable(torch.from_numpy(lam)).to(device)
        lam_adv = torch.clamp(lam_adv, 0, 1)  # clamp to range [0,1)
        lam_adv.requires_grad = True

        index = torch.randperm(N).cuda()
        # initialize x_mix
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)

        # Weight and Data Ascent Step
        output1 = model(raw_data)
        output2 = model(mix_inputs)
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        grad_lam_adv = lam_adv.grad.data
        grad_norm = torch.norm(grad_lam_adv, p=2) + 1.e-16
        lam_adv.data.add_(grad_lam_adv * 0.05 / grad_norm)  # gradient assend by SAM
        lam_adv = torch.clamp(lam_adv, 0, 1)
        optimizer.perturb_step()

        # Weight Descent Step
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)
        mix_inputs = mix_inputs.detach()

        lam_adv = lam_adv.detach()
        output1 = model(raw_data)
        output2 = model(mix_inputs)
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        optimizer.step()
def train_projected (args, model,device,x,y,optimizer,criterion,feature_mat, inner_steps=2):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()

    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b].view(-1, 28*28)
        raw_data, raw_target = data.to(device), y[b].to(device)

        # Data Perturbation Step
        # initialize lamb mix:
        N = data.shape[0]
        lam = (beta_distributions(size=N, alpha=args.mixup_alpha)).astype(np.float32)
        lam_adv = Variable(torch.from_numpy(lam)).to(device)
        lam_adv = torch.clamp(lam_adv, 0, 1)  # clamp to range [0,1)
        lam_adv.requires_grad = True

        index = torch.randperm(N).cuda()
        # initialize x_mix
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)

        # Weight and Data Ascent Step
        output1 = model(raw_data)
        output2 = model(mix_inputs)
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        grad_lam_adv = lam_adv.grad.data
        grad_norm = torch.norm(grad_lam_adv, p=2) + 1.e-16
        lam_adv.data.add_(grad_lam_adv * 0.05 / grad_norm)  # gradient assend by SAM
        lam_adv = torch.clamp(lam_adv, 0, 1)
        optimizer.perturb_step()

        # Weight Descent Step
        mix_inputs, mix_targets_a, mix_targets_b = aug_model(raw_data, lam_adv, raw_target, index)
        mix_inputs = mix_inputs.detach()
        lam_adv = lam_adv.detach()
        output1 = model(raw_data)
        output2 = model(mix_inputs)
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a,
                                                                                    mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        # Gradient Projections 
        for k, (m,params) in enumerate(model.named_parameters()):
            sz = params.grad.data.size(0)
            params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), feature_mat[k]).view(params.size())
        optimizer.step()
def test (args, model, device, x, y, criterion):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)

    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b].view(-1,28*28)
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc
def get_representation_matrix (net, device, x, y=None):
    # Collect activations by forward pass
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:300] # Take random training samples
    example_data = x[b].view(-1,28*28)
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    batch_list=[300, 300, 300]
    mat_list=[] # list contains representation matrix of each layer
    act_key=list(net.act.keys())

    for i in range(len(act_key)):
        bsz=batch_list[i]
        act = net.act[act_key[i]].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)

    log.info('-'*30)
    log.info('Representation Matrix')
    log.info('-'*30)
    for i in range(len(mat_list)):
        log.info ('Layer {} : {}'.format(i+1,mat_list[i].shape))
    log.info('-'*30)
    return mat_list    
def update_GradientMemory (model, mat_list, threshold, feature_list=[],):
    log.info ('Threshold: ', threshold) 
    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1**2).sum()
            act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
            U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
            # criteria (Eq-9)
            sval_hat = (S**2).sum()
            sval_ratio = (S**2)/sval_total               
            accumulated_sval = (sval_total-sval_hat)/sval_total
            
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                log.info ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    log.info('-'*40)
    log.info('Gradient Constraints Summary')
    log.info('-'*40)
    for i in range(len(feature_list)):
        log.info ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    log.info('-'*40)
    return feature_list  
def main(args):
    tstart=time.time()
    ## Device Setting 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## Load PMNIST DATASET
    from dataloader import pmnist as pmd
    data, taskcla, inputsize=pmd.get(seed=args.seed, pc_valid=args.pc_valid)

    acc_matrix = np.zeros((10, 10))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        threshold = np.array([args.gpm_thro1, args.gpm_thro2, args.gpm_thro3])
        log.info('threshold:' + str(threshold))

        log.info('*'*100)
        log.info('Task {:2d} ({:s})'.format(k,data[k]['name']))
        log.info('*'*100)
        xtrain = data[k]['train']['x']
        ytrain = data[k]['train']['y']
        xvalid = data[k]['valid']['x']
        yvalid = data[k]['valid']['y']
        xtest = data[k]['test']['x']
        ytest = data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        log.info ('-'*40)
        log.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        log.info ('-'*40)
        
        if task_id == 0:
            model = MLPNet(args.n_hidden, args.n_outputs).to(device)
            log.info ('Model parameters ---')
            for k_t, (m, param) in enumerate(model.named_parameters()):
                log.info (k_t,m,param.shape)
            log.info ('-'*40)

            feature_list =[]
            base_optimizer = optim.SGD(model.parameters(), lr=lr)
            optimizer = SAM(base_optimizer, model)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, tr_loss,tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                log.info('')
            # Test
            log.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            feature_list = update_GradientMemory (model, mat_list, threshold, feature_list)

        else:
            base_optimizer = optim.SGD(model.parameters(), lr=lr)
            optimizer = SAM(base_optimizer, model)
            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(model.act)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                log.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            log.info ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,  criterion)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch, tr_loss, tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                log.info('')
            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))  
            # Memory Update
            mat_list = get_representation_matrix (model, device, xtrain, ytrain)
            feature_list = update_GradientMemory (model, mat_list, threshold, feature_list)

        # save accuracy 
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion) 
            jj +=1
        log.info('Accuracies =')
        for i_a in range(task_id + 1):
            acc_ = ''
            for j_a in range(acc_matrix.shape[1]):
                acc_ += '{:5.1f}% '.format(acc_matrix[i_a, j_a])
            log.info(acc_)

        # update task id 
        task_id +=1

    log.info('-'*50)
    # Simulation Results 
    log.info ('Task Order : {}'.format(np.array(task_list)))
    log.info ('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean())) 
    bwt=np.mean((acc_matrix[-1]-np.diag(acc_matrix))[:-1]) 
    log.info ('Backward transfer: {:5.2f}%'.format(bwt))
    log.info('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    log.info('-'*50)
    return acc_matrix[-1].mean(), bwt
def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

if __name__ == "__main__":
    # Training parameters
    parser = argparse.ArgumentParser(description='Sequential PMNIST with DFGP')
    parser.add_argument('--batch_size_train', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5, metavar='N',
                        help='number of training epochs/task (default: 5)')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 2)')
    parser.add_argument('--pc_valid',default=0.1,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-5, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=6, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=2, metavar='LRF',
                        help='lr decay factor (default: 2)')
    # Architecture
    parser.add_argument('--n_hidden', type=int, default=100, metavar='NH',
                        help='number of hidden units in MLP (default: 100)')
    parser.add_argument('--n_outputs', type=int, default=10, metavar='NO',
                        help='number of output units in MLP (default: 10)')
    parser.add_argument('--n_tasks', type=int, default=10, metavar='NT',
                        help='number of tasks (default: 10)')
    parser.add_argument('--savename', type=str, default='./logs/PMNIST/',
                        help='save path')
    parser.add_argument('--gpm_thro1', type=float, default=0.95, metavar='THR1',
                        help='projection thr1')
    parser.add_argument('--gpm_thro2', type=float, default=0.99, metavar='THR2',
                        help='projection thr2')
    parser.add_argument('--gpm_thro3', type=float, default=0.99, metavar='THR3',
                        help='projection thr3')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))

    for thro_1 in [0.94, 0.95, 0.96]:
        for thro_2_and_3 in [0.96, 0.97, 0.98, 0.99]:
            for mixup_weight in [0.01, 0.001, 0.0001]:

                accs, bwts = [], []
                str_time = str_time_ + '_' + str(thro_1) + '_' + str(thro_2_and_3) + '_' + str(thro_2_and_3)

                args.mixup_weight = mixup_weight
                args.gpm_thro1 = thro_1
                args.gpm_thro2 = thro_2_and_3
                args.gpm_thro3 = thro_2_and_3

                for seed_ in [1, 2]:
                    try:
                        args.seed = seed_
                        log.info('=' * 100)
                        log.info('Arguments =')
                        log.info(str(args))
                        log.info('=' * 100)

                        acc, bwt = main(args)
                        accs.append(acc)
                        bwts.append(bwt)
                    except:
                        print("seed " + str(seed_) + "Error!!")



