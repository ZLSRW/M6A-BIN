import json
from datetime import datetime
import warnings

from data_loader.SiteBinding_dataloader1 import ForecastDataset
from .IBGNN import *
# from models.seq_graph import Model

import torch.utils.data as torch_data
import time
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from .Utils import *
from .Fusion_Classifier import *
from .configure import *

warnings.filterwarnings("ignore")

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def save_model(model, model_dir, epoch, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, str(fold)+'_'+epoch + '_PepBindA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def save_model1(model, model_dir, epoch, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, str(fold)+'_'+'best'+ '_IBGGN.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir,i):
    if not model_dir:
        return
    file_name = os.path.join(model_dir, str(i)+'_best_IBGGN.pt')
    print(file_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model



def validate_inference_binding_site(model,Graphrepresentation1,Graphrepresentation2,Graphrepresentation3,GraphrepresentationFusion,Classifier,dataloader):
    model.eval()
    Graphrepresentation1.eval()
    Graphrepresentation2.eval()
    Graphrepresentation3.eval()
    GraphrepresentationFusion.eval()
    Classifier.eval()

    with torch.no_grad():
        for i, ( graph,labels,onehot_graph,onehots,onehot_feature) in enumerate(dataloader):

            Temp_x1_feature=[]
            Temp_x2_feature=[]
            graph = graph.to('cuda:0')  # 输入cgr序列
            labels=labels.to('cuda:0')
            onehot_graph=onehot_graph.to('cuda:0')
            onehot_x = onehots.to('cuda:0')  # 输入结合位点
            onehot_x=onehot_x.reshape(-1,41,4)
            onehot_feature = onehot_feature.to('cuda:0')  # 输入结合位点
            # inputs = normalized_input(inputs)
            # print(inputs.shape)  #torch.Size([32, 12, 51])
            # print(inputs_labels.shape)  # 32x12

            x1, x2,_,_,_,_,_= model(graph,onehot_graph,onehot_x,onehot_feature,onehot_x,onehot_feature,False)

            h1,x1_prob=Graphrepresentation1(x1)
            h2,x2_prob=Graphrepresentation2(x2)
            ELMo_feature,ELMo_prob=Graphrepresentation3(onehot_feature)
            # R=GraphrepresentationFusion(h1,h2)+ELMo_feature
            R=GraphrepresentationFusion(h1,h2)
            # R=R+ELMo_feature
            # R=ELMo_feature
            labels_pred=Classifier(R)
            labels_pred.squeeze()

            result,Real_Prediction,Real_Prediction_Prob=Indicator(labels,labels_pred)

            #
            validate_auc, _, _ = auroc(labels_pred, labels)
            validate_aupr, _, _ = auprc(labels_pred, labels)
            # print('validate_auc: '+str(validate_auc)+' '+'validate_aupr: '+str(validate_aupr))
            result[2]=round(validate_aupr,4)

            # x1,x2保存，softmax的结果保存，最后一维为对应的权重
            x1 = torch.cat((x1, x1_prob), dim=-1)
            x2 = torch.cat((x2, x2_prob), dim=-1)

            x1 = list(x1.cpu().detach().numpy())
            x2 = list(x2.cpu().detach().numpy())

            Temp_x1_feature.extend(x1)
            Temp_x2_feature.extend(x2)

            labels_real = list(labels.contiguous().view(-1).cpu().detach().numpy())
            forecast_features = list(R.cpu().detach().numpy()) #全局特征
            xx = 0
            while xx < len(forecast_features):
                forecast_features[xx]=list(forecast_features[xx])
                forecast_features[xx].append(int(labels_real[xx]))
                xx += 1

            onehot_feature=list(onehot_feature.cpu().detach().numpy())

    return  result,forecast_features,Real_Prediction,Real_Prediction_Prob,Temp_x1_feature,Temp_x2_feature,onehot_feature

def train(train_data, valid_data, args,result_file,fold,species):
    # node_cnt = int((train_data.shape[1]-3)/3) #100 (固定窗口大小或自适应窗口大小)
    ISGNN = InvertibleBigraphNeuralNetwork(batch=args.batch_size,batch1=args.batch_size1,size=args.size,num=args.num) #输入三个参数

    #初始化图表示模块、融合模块和分类模块
    graphRepresentation1=GraphRepresentation(256, 256,device='cuda:0')
    graphRepresentation2=GraphRepresentation(256, 256,device='cuda:0')
    graphRepresentation3=GraphRepresentation(256, 256,device='cuda:0')

    graphRepresentationFusion=GraphRepresentationfusion(device='cuda:0')
    classifier=Classifier(64,64,device='cuda:0')

    ISGNN.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=ISGNN.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=ISGNN.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data)
    valid_set = ForecastDataset(valid_data)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    # 对训练集里的每个类别加一个权重。如果该类别的样本数多，那么它的权重就低，反之则权重就高
    criterion = torch.nn.BCELoss( reduction='mean')  # 计算目标值和预测值之间的二进制交叉熵损失函数

    total_params = 0
    for name, parameter in ISGNN.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_Acc= 0.0
    best_result=[]
    best_Real_Predition=[]
    best_Real_Predition_Prob=[]

    best_train_feature=[]
    best_x1_feature=[]
    best_x2_feature=[]

    best_validate_feature=[]
    best_initial_feature=[]
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        ISGNN.train()
        loss_total = 0
        cnt = 0
        auc_total=0
        aupr_total=0

        Temp_train_feature=[]
        Temp_x1_feature=[]
        Temp_x2_feature=[]

        for i, (
        graph,labels,onehot_graph,onehots,onehot_feature) in enumerate(train_loader):
            graph = graph.to(args.device)  # 输入cgr序列
            labels=labels.to(args.device)
            # print(torch.sum(labels))
            onehot_graph=onehot_graph.to(args.device)
            onehot_x = onehots.to(args.device)  # 标签图节点初始表征
            onehot_x=onehot_x.reshape(-1,41,4)
            Elom4_feature = onehot_feature.to(args.device)  # 结构图节点表征

            # print(onehot_x.shape) # torch.Size([706, 41, 4])
            # print(onehot_graph.shape)
            # print(onehot_feature.shape) #one-hot表征，torch.Size([705, 41, 256])
            # print(labels)

            # x1,x2需要保留，softmax也需要保留
            x1, x2, Loss1, Loss2, Loss3, Loss4, Loss5= ISGNN(graph,onehot_graph,onehot_x,Elom4_feature,onehot_x,Elom4_feature,False) #输入为两个图，两个图的节点初始特征，以及逆过程的
            # print(x1.shape) #torch.Size([706, 41, 64])
            # print(x2.shape) #torch.Size([706, 41, 64])

            # fusion and classifier
            h1,h1_prob=graphRepresentation1(x1)
            h2,h2_prob=graphRepresentation2(x2)
            Elom4_feature,Elom4_prob=graphRepresentation3(Elom4_feature)
            # Elom4,h2_prob=graphRepresentation(x2,Elom4_feature)



            # R=graphRepresentationFusion(h1,h2)+Elom4_feature
            R=graphRepresentationFusion(h1,h2)
            # R=R+Elom4_feature
            # R=R

            labels_pred=classifier(R)
            labels_pred.squeeze()

            # x1,x2保存，softmax的结果保存，最后一维为对应的权重
            x1 = torch.cat((x1, h1_prob), dim=-1)
            x2 = torch.cat((x2, h2_prob), dim=-1)

            x1 = list(x1.cpu().detach().numpy())
            x2 = list(x2.cpu().detach().numpy())

            Temp_x1_feature.extend(x1)
            Temp_x2_feature.extend(x2)

            # 最后的训练特征保存,带标签
            labels_real = list(labels.contiguous().view(-1).cpu().detach().numpy())
            forecast_feature = list(R.cpu().detach().numpy())
            xx = 0
            while xx < len(forecast_feature):
                forecast_feature[xx]=list(forecast_feature[xx])
                forecast_feature[xx].append(int(labels_real[xx]))
                xx += 1
            Temp_train_feature.extend(forecast_feature)

            train_auc,_,_=auroc(labels_pred.squeeze(),labels)
            train_aupr,_,_=auprc(labels_pred.squeeze(),labels)
            # print(labels_pred)
            # print(labels)
            binding_loss = criterion(labels_pred.squeeze(), labels.float())
            all_loss=binding_loss+Loss1+Loss2+Loss4+Loss5
            # all_loss=binding_loss+Loss1+Loss2+Loss3+(Loss4+Loss5)
            # all_loss=binding_loss+0.001*Loss1+0.001*Loss2+0.0001*Loss3+0.00001*Loss4+0.0000001*Loss5
            # all_loss=binding_loss
            # all_loss=binding_loss+0.001*Loss1+0.001*Loss2+0.001*Loss3+0.00001*Loss4+0.00001*Loss5
            # all_loss=binding_loss+0.001*Loss1+0.001*Loss2+0.0001*Loss3+0.00001*Loss4+0.0000001*Loss5 #Loss5需要更小的系数，加性耦合会使得元素值加大
            # all_loss=binding_loss

            auc_total+=train_auc
            aupr_total+=train_aupr

            #训练过程中
            """
            loss需要进行修改，不仅要考虑forecast和target，还要考虑预测结合位点和实际结合位点的关系（结合位点的损失不区分输入和目标，而是一起考虑）；
            """
            # print('reconstuction_loss '+str(reconstuction_loss)+' '+'train_auc '+str(train_auc))
            # print('epoch %d, binding_loss %.4f, Loss1 %.4f, Loss2 %.4f, Loss3 %.4f, Loss4 %.4f, Loss5 %.4f, train_auc %.4f, train_aupr %.4f  '
            #       % (epoch + 1, all_loss, 0.001*Loss1, 0.001*Loss2, 0.0001*Loss3, 0.00001*Loss4, 0.00001*Loss5, train_auc, train_aupr))
            print('epoch %d, binding_loss %.4f, Loss1 %.4f, Loss2 %.4f, Loss3 %.4f, Loss4 %.4f, Loss5 %.4f, train_auc %.4f, train_aupr %.4f  '
                  % (epoch + 1, all_loss, Loss1, Loss2, Loss3, Loss4, Loss5, train_auc, train_aupr))
            cnt += 1

            # loss.backward()
            ISGNN.zero_grad()

            all_loss.backward()

            # my_optim.step()
            my_lr_scheduler.step()

            loss_total += float(all_loss)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | train_auc {:5.4f}| train_aupr {:5.4f}'.format(epoch+1, (
                time.time() - epoch_start_time), loss_total / cnt,auc_total/cnt,aupr_total/cnt))
        # print('| end of epoch {:3d} | time: {:5.2f}s | train_auc1 {:5.4f} | train_aupr1 {:5.4f}|train_auc2 {:5.4f} | train_aupr2 {:5.4f}|train_auc3 {:5.4f} | train_aupr3 {:5.4f}|train_auc4 {:5.4f} | train_aupr4 {:5.4f}| '.format(epoch+1, (
        #         time.time() - epoch_start_time), auc_total_bag[0]/cnt, aupr_total_bag[0]/cnt, auc_total_bag[1]/cnt, aupr_total_bag[1]/cnt, auc_total_bag[2]/cnt, aupr_total_bag[2]/cnt, auc_total_bag[3]/cnt, aupr_total_bag[3]/cnt))
        # if (epoch+1)%10==0:
        #     save_model(IBGNN, result_file, epoch, fold)

        if 1==1:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')

            result,validate_features,Real_prediction,Real_prediction_prob,valid_x1_feature,valid_x2_feature,initial_feature=validate_inference_binding_site(ISGNN,graphRepresentation1,graphRepresentation2,graphRepresentation3,graphRepresentationFusion,classifier, valid_loader)

            MCC = result[0]
            auc = result[1]
            aupr=result[2]
            F1 = result[3]
            Acc = result[4]
            Sen = result[5]
            Spec = result[6]
            Prec = result[7]

            print('validate_MCC: '+str(round(MCC,4))+' '+' validate_auc: '+str(round(auc,4))+' validate_aupr: '+str(round(aupr,4))+' '+' validate_F1: '+str(round(F1,4))+' '+
                  ' validate_Acc: '+str(round(Acc,4))+' '+' validate_Sen: '+str(round(Sen,4))+' '+' validate_Spec: '+str(round(Spec,4))+' '
                   +' validate_Prec: '+str(round(Prec,4)))


            if Acc >= best_validate_Acc:
                best_validate_Acc = Acc
                best_result=result
                best_Real_Predition=Real_prediction
                best_Real_Predition_Prob=Real_prediction_prob
                best_train_feature=Temp_train_feature

                best_x1_feature=Temp_x1_feature
                best_x2_feature=Temp_x2_feature

                best_validate_feature=validate_features

                best_initial_feature=initial_feature
                is_best_for_now = True

                # save_model1(ISGNN, result_file, epoch, fold)
        # best_train_feature = Temp_train_feature但是
        # best_validate_feature = validate_features

        # pd.DataFrame(forecast_feature.detach().numpy()).to_csv('Train_Test_final_feature'+str(epoch)+'.csv')
    # All_result.append(best_result)
    #全局表示的各项指标，真实值_预测值，真实值_预测概率，
    # StorFile(All_result, './utils/0.RNA_m_process/m6A/M41_dependcy/Local/All_result' + str(fold) + '.csv')

    # StorFile(best_Real_Predition, './Pre-Encoding/data/'+str(species)+'/Result/Real_Predition'+str(fold)+'.csv')
    #
    # StorFile(best_Real_Predition_Prob, './Pre-Encoding/data/'+str(species)+'/Result/Real_Predition_prob'+str(fold)+'.csv')
    #
    # StorFile(best_train_feature,'./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/Train_feature' + str(fold) + '.csv')
    #
    # StorFile(best_validate_feature,'./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/Validate_feature' + str(fold) + '.csv')
    #
    # np.save('./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/valid_x1_feature' + str(fold) + '.npy',valid_x1_feature)
    # np.save('./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/valid_x2_feature' + str(fold) + '.npy',valid_x2_feature)
    #
    # np.save('./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/train_x1_feature' + str(fold) + '.npy',best_x1_feature)
    # np.save('./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/train_x2_feature' + str(fold) + '.npy',best_x2_feature)
    #
    # np.save('./Pre-Encoding/data/'+str(species)+'/Train_Test_Feature/initial_feature' + str(fold) + '.npy',best_initial_feature)

    print(
        'best_MCC: ' + str(round(best_result[0], 4)) + ' ' + ' best_auc: ' + str(round(best_result[1], 4)) + ' best_aupr: ' + str(
            round(best_result[2], 4)) + ' ' + ' best_F1: ' + str(round(best_result[3], 4)) + ' ' +
        ' best_Acc: ' + str(round(best_result[4], 4)) + ' ' + ' best_Sen: ' + str(
            round(best_result[5], 4)) + ' ' + ' best_Spec: ' + str(round(best_result[6], 4)) + ' '
        + ' best_Prec: ' + str(round(best_result[7], 4)))

    return forecast_feature,best_result

def inverse_validate_process(args, result_train_file,x1,x2,pre1,pre2,i): #五个参数分别为：模型、输出的两种表征、输入的两种表征
    model = load_model(result_train_file,i)
    x1=np.array(x1,dtype='float64')
    x1=torch.from_numpy(x1).type(torch.float).to(args.device)

    x2=np.array(x2,dtype='float64')
    x2=torch.from_numpy(x2).type(torch.float).to(args.device)

    pre1=np.array(pre1,dtype='float64')
    pre1=torch.from_numpy(pre1).type(torch.float).to(args.device)

    pre2=np.array(pre2,dtype='float64')
    pre2=torch.from_numpy(pre2).type(torch.float).to(args.device)


    model.eval()
    G1=None
    G2=None
    G1, G2= model(G1, G2, x1, x2, pre1, pre2,i,True)
    return  G1,G2


def test(test_data, args, result_train_file,i): #

    model = load_model(result_train_file,i)
    # print(model)

    valid_set = ForecastDataset(test_data)

    test_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result,feature,Real_Prediction,Real_Prediction_Prob,motifs,xtypes=validate_inference_binding_site(model, test_loader,)

    # motifs = motifs.detach().numpy()
    # new_motifs = []
    # new_motifs.extend([mo] for mo in motifs)
    # x_type = x_type.detach().numpy()

    MCC = result[0]
    auc = result[1]
    aupr = result[2]
    F1 = result[3]
    Acc = result[4]
    Sen = result[5]
    Spec = result[6]
    Prec = result[7]
    print(
        'validate_MCC: ' + str(round(MCC, 4)) + ' ' + ' validate_auc: ' + str(round(auc, 4)) + ' validate_aupr: ' + str(
            round(aupr, 4)) + ' ' + ' validate_F1: ' + str(round(F1, 4)) + ' ' +
        ' validate_Acc: ' + str(round(Acc, 4)) + ' ' + ' validate_Sen: ' + str(
            round(Sen, 4)) + ' ' + ' validate_Spec: ' + str(round(Spec, 4)) + ' '
        + ' validate_Prec: ' + str(round(Prec, 4)))
    # import pandas as pd
    # pd.DataFrame(result).to_csv('./Motifis_analysis/Score_motif/Mouse_kidney/result0.csv')
    # StorFile(Real_Prediction, './Motifis_analysis/Score_motif/Mouse_kidney/Real_Prediction0.csv')
    # StorFile(Real_Prediction_Prob, './Motifis_analysis/Score_motif/Mouse_kidney/Real_Prediction_prob0.csv')
    # StorFile(new_motifs, './Motifis_analysis/Score_motif/Mouse_kidney/motif.csv')
    # StorFile(x_type, './Motifis_analysis/Score_motif/Mouse_kidney/x_type.csv')

    import pandas as pd
    # pd.DataFrame(result).to_csv('./Case_CS_CT/CS/R_M/result'+str(i)+'.csv')
    # StorFile(Real_Prediction, './Case_CS_CT/CS/R_M/Real_Prediction'+str(i)+'.csv')
    # StorFile(Real_Prediction_Prob, './Case_CS_CT/CS/R_M/Real_Prediction_prob'+str(i)+'.csv')
    # StorFile(feature, './Case_CS_CT/CS/R_M/feature'+str(i)+'.csv')
    #
    # pd.DataFrame(motifs).to_csv('./Case_CS_CT/CS/R_M/motif' + str(i) + '.csv')
    # pd.DataFrame(xtypes).to_csv('./Case_CS_CT/CS/R_M/type' + str(i) + '.csv')

    pd.DataFrame(result).to_csv('./Case_CS_CT/CT/k_b/result'+str(i)+'.csv')
    StorFile(Real_Prediction, './Case_CS_CT/CT/k_b/Real_Prediction'+str(i)+'.csv')
    StorFile(Real_Prediction_Prob, './Case_CS_CT/CT/k_b/Real_Prediction_prob'+str(i)+'.csv')
    StorFile(feature, './Case_CS_CT/CT/k_b/feature'+str(i)+'.csv')


    return

