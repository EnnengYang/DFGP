import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable
import os
import os.path
from collections import OrderedDict
import numpy as np
import argparse,time
from copy import deepcopy
import time
from flatness_minima import SAM

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 2)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 9, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 84, 84)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 84, 84))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y
def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)
def get_model(model):
    return deepcopy(model.state_dict())
def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return
def adjust_learning_rate(optimizer, epoch, args):
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= args.lr_factor  
def beta_distributions(size, alpha=1):
    return np.random.beta(alpha, alpha, size=size)
class AugModule(nn.Module):
    def __init__(self):
        super(AugModule, self).__init__()
    def forward(self, xs, lam, y, index):
        x_ori = xs
        N = x_ori.size()[0]
        x_ori_perm = x_ori[index, :]
        lam = lam.view((N, 1, 1, 1)).expand_as(x_ori)
        x_mix = (1 - lam) * x_ori + lam * x_ori_perm
        y_a, y_b = y, y[index]
        return x_mix, y_a, y_b
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    loss_a = lam * criterion(pred, y_a)
    loss_b = (1 - lam) * criterion(pred, y_b)
    return loss_a.mean() + loss_b.mean()
def train(args, model, device, x,y, optimizer,criterion, task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
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
        output1 = model(raw_data)[task_id]
        output2 = model(mix_inputs)[task_id]
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

        output1 = model(raw_data)[task_id]
        output2 = model(mix_inputs)[task_id]
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        # Update
        optimizer.step()
def train_projected(args,model,device,x,y,optimizer,criterion,feature_mat,task_id):
    model.train()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    aug_model = AugModule()
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
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
        output1 = model(raw_data)[task_id]
        output2 = model(mix_inputs)[task_id]
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

        output1 = model(raw_data)[task_id]
        output2 = model(mix_inputs)[task_id]
        loss = criterion(output1, raw_target) + args.mixup_weight * mixup_criterion(criterion, output2, mix_targets_a, mix_targets_b, lam_adv.detach())
        loss.backward()
        optimizer.unperturb_step()

        # Gradient Projections
        kk = 0 
        for k, (m,params) in enumerate(model.named_parameters()):
            if len(params.size())==4 and 'weight' in m:
                sz =  params.grad.data.size(0)
                params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1), feature_mat[kk]).view(params.size())
                kk+=1
            elif len(params.size())==1 and task_id !=0:
                params.grad.data.fill_(0)

        # Update
        optimizer.step()
def test(args, model, device, x, y, criterion, task_id):
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
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)
            pred = output[task_id].argmax(dim=1, keepdim=True) 
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc
def get_representation_matrix_ResNet18 (net, device, x, y=None): 
    # Collect activations by forward pass
    net.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:100] # ns=100 examples 
    example_data = x[b]
    example_data = example_data.to(device)
    example_out  = net(example_data)
    
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

    batch_list  = [10,10,10,10,10,10,10,10,50,50,50,100,100,100,100,100,100] #scaled
    # network arch 
    stride_list = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    map_list = [84, 42, 42, 42, 42, 42, 21, 21, 21, 21, 11, 11, 11, 11, 6, 6, 6]
    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160]

    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[] # list containing GPM Matrices 
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3 
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat) 

    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1

    log.info('-'*30)
    log.info('Representation Matrix')
    log.info('-'*30)
    for i in range(len(mat_final)):
        log.info ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    log.info('-'*30)
    return mat_final    
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
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ## Load miniimagenet DATASET
    from dataloader import miniimagenet as data_loader
    dataloader = data_loader.DatasetGen(args)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    acc_matrix=np.zeros((20, 20))
    criterion = torch.nn.CrossEntropyLoss()

    task_id = 0
    task_list = []
    for k,ncla in taskcla:
        threshold = np.array([args.gpm_thro] * 20)
        data = dataloader.get(k)

        log.info('*'*100)
        log.info('Task {:2d} ({:s})'.format(k, data[k]['name']))
        log.info('*'*100)
        xtrain=data[k]['train']['x']
        ytrain=data[k]['train']['y']
        xvalid=data[k]['valid']['x']
        yvalid=data[k]['valid']['y']
        xtest =data[k]['test']['x']
        ytest =data[k]['test']['y']
        task_list.append(k)

        lr = args.lr 
        best_loss=np.inf
        log.info ('-'*40)
        log.info ('Task ID :{} | Learning Rate : {}'.format(task_id, lr))
        log.info ('-'*40)
        
        if task_id==0:
            model = ResNet18(taskcla,20).to(device) # base filters: 20
 
            best_model=get_model(model)
            feature_list =[]
            base_optimizer = optim.SGD(model.parameters(), lr=lr)
            optimizer = SAM(base_optimizer, model)

            for epoch in range(1, args.n_epochs+1):
                # Train
                clock0=time.time()
                train(args, model, device, xtrain, ytrain, optimizer, criterion, k)
                clock1=time.time()
                tr_loss,tr_acc = test(args, model, device, xtrain, ytrain,  criterion, k)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                            tr_loss,tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, k)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer.optimizer, epoch, args)
            set_model_(model,best_model)
            # Test
            log.info ('-'*40)
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, k)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update  
            mat_list = get_representation_matrix_ResNet18 (model, device, xtrain, ytrain)
            feature_list = update_GradientMemory (model, mat_list, threshold, feature_list)
        else:
            base_optimizer = optim.SGD(model.parameters(), lr=lr)
            optimizer = SAM(base_optimizer, model)

            feature_mat = []
            # Projection Matrix Precomputation
            for i in range(len(feature_list)):
                Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).to(device)
                log.info('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                feature_mat.append(Uf)
            log.info ('-'*40)
            for epoch in range(1, args.n_epochs+1):
                # Train 
                clock0=time.time()
                train_projected(args, model,device,xtrain, ytrain,optimizer,criterion,feature_mat,k)
                clock1=time.time()
                tr_loss, tr_acc = test(args, model, device, xtrain, ytrain,criterion,k)
                log.info('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | time={:5.1f}ms |'.format(epoch,\
                                                        tr_loss, tr_acc, 1000*(clock1-clock0)))
                # Validate
                valid_loss,valid_acc = test(args, model, device, xvalid, yvalid, criterion,k)
                log.info(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc))
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=get_model(model)
                    patience=args.lr_patience
                else:
                    patience-=1
                    if patience<=0:
                        lr/=args.lr_factor
                        log.info(' lr={:.1e}'.format(lr))
                        if lr<args.lr_min:
                            break
                        patience=args.lr_patience
                        adjust_learning_rate(optimizer.optimizer, epoch, args)
            set_model_(model,best_model)
            # Test 
            test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion,k)
            log.info('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
            # Memory Update 
            mat_list = get_representation_matrix_ResNet18 (model, device, xtrain, ytrain)
            feature_list = update_GradientMemory (model, mat_list, threshold, feature_list)

        # save accuracy
        jj = 0 
        for ii in np.array(task_list)[0:task_id+1]:
            xtest =data[ii]['test']['x']
            ytest =data[ii]['test']['y'] 
            _, acc_matrix[task_id,jj] = test(args, model, device, xtest, ytest,criterion,ii) 
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
    parser = argparse.ArgumentParser(description='miniimagenet datasets with DFGP')
    parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',
                        help='number of training epochs/task (default: 100)')
    parser.add_argument('--seed', type=int, default=37, metavar='S',
                        help='random seed (default: 37)')
    parser.add_argument('--pc_valid',default=0.02,type=float,
                        help='fraction of training data used for validation')
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--lr_min', type=float, default=1e-3, metavar='LRM',
                        help='minimum lr rate (default: 1e-5)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 6)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 2)')
    parser.add_argument('--savename', type=str, default='./logs/MINI/',
                        help='save path')
    parser.add_argument('--gpm_thro', type=float, default=0.97, metavar='gradient projection',
                        help='gpm_thro')
    parser.add_argument('--mixup_alpha', type=float, default=20, metavar='Alpha',
                        help='mixup_alpha')
    parser.add_argument('--mixup_weight', type=float, default=0.1, metavar='Weight',
                        help='mixup_weight')

    args = parser.parse_args()
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, 'log_{}.txt'.format(str_time_))

    for mixup_weight in [0.0001, 0.01, 0.05, 0.1]:

        accs, bwts = [], []
        str_time = str_time_ + '_' + str(mixup_weight)
        args.mixup_weight = mixup_weight

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




