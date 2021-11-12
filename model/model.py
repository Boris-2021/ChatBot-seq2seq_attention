#导入库
import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib import ticker
from nltk.translate.bleu_score import sentence_bleu
import time,random,os,jieba,logging
import numpy as np
import pandas as pd
jieba.setLogLevel(logging.INFO)
# 不显示debug信息，只要大于 logging.DEBUG 级别即可

# 定义开始符和结束符
sosToken = 1
eosToken = 0

#  调试用的导包
# import sys
# print(sys.path)
# sys.path.append('C:\\Users\\boris\\Desktop\\文字聊天机器人指导\\00.配套资料（代码）\\nlp_chatbot\\8\\model\\pre_process')
from model.pre_process import *


#定义Encoder
class EncoderRNN(nn.Module):
    #初始化
    def __init__(self, featureSize, hiddenSize, embedding, numLayers=1, dropout=0.1, bidirectional=True):
        '''
        :param featureSize: 特征大小
        :param hiddenSize: GRU, lstm 隐层大小
        :param embedding:Embedding就是从原始数据提取出来的Feature，也就是那个通过神经网络映射之后的低维向量。
        :param numLayers:
        :param dropout:防止过拟合
        :param bidirectional:BI双向GRU
        '''
        super(EncoderRNN, self).__init__()
        self.embedding = embedding
        #核心API，建立双向GRU
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), bidirectional=bidirectional, batch_first=True)
        #超参
        self.featureSize = featureSize  # 特征
        self.hiddenSize = hiddenSize # 隐层
        self.numLayers = numLayers # 一个rnn内部几层网络
        self.bidirectional = bidirectional

    #前向计算，训练和测试必须的部分
    def forward(self, input, lengths, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        #给定输入
        input = self.embedding(input) # => batchSize × seq_len × feaSize
        #加入paddle 方便计算
        packed = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hn = self.gru(packed, hidden) # output: batchSize × seq_len × hiddenSize*d; hn: numLayers*d × batchSize × hiddenSize 
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #确定是否是双向GRU
        if self.bidirectional:
            output = output[:,:,:self.hiddenSize] + output[:,:,self.hiddenSize:]
        return output, hn


#定义Decoder
class DecoderRNN(nn.Module):
    #初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        #核心API
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, batch_first=True)
        self.out = nn.Linear(featureSize, outputSize)
    #定义前向计算
    def forward(self, input, hidden):
        # input: batchSize × seq_len; hidden: numLayers*d × batchSize × hiddenSize
        input = self.embedding(input) # => batchSize × seq_len × feaSize
        #relu激活，softmax计算输出
        input = F.relu(input)
        output,hn = self.gru(input, hidden) # output: batchSize × seq_len × feaSize; hn: numLayers*d × batchSize × hiddenSize
        output = F.log_softmax(self.out(output), dim=2) # output: batchSize × seq_len × outputSize
        return output,hn,torch.zeros([input.size(0), 1, input.size(1)])


#定义 BahdanauAttention的Decoder
class BahdanauAttentionDecoderRNN(nn.Module):
    #初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1):
        super(BahdanauAttentionDecoderRNN, self).__init__()
        self.embedding = embedding
        #定义attention的权重还有如何联合，及dropout，防止过拟合
        self.dropout = nn.Dropout(dropout)
        self.attention_weight = nn.Linear(hiddenSize*2, 1)
        # 定义如何进行链接
        self.attention_combine = nn.Linear(featureSize+hiddenSize, featureSize)
        #核心API 搭建GRU层，并给定超参
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), batch_first=True)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.numLayers = numLayers
    #定义前向计算
    def forward(self, inputStep, hidden, encoderOutput):
        #input做了dropout的操作，主要是防止过拟合
        inputStep = self.embedding(inputStep) # => batchSize × 1 × feaSize
        inputStep = self.dropout(inputStep)
       #计算attention的权重部分，attention的本质是softmax
        attentionWeight = F.softmax(self.attention_weight(torch.cat((encoderOutput, hidden[-1:].expand(encoderOutput.size(1),-1,-1).transpose(0,1)), dim=2)).transpose(1,2), dim=2)

        context = torch.bmm(attentionWeight, encoderOutput) # context: batchSize × 1 × hiddenSize
        # 联合
        attentionCombine = self.attention_combine(torch.cat((inputStep, context), dim=2)) # attentionCombine: batchSize × 1 × feaSize
        attentionInput = F.relu(attentionCombine) # attentionInput: batchSize × 1 × feaSize
        output, hidden = self.gru(attentionInput, hidden) # output: batchSize × 1 × hiddenSize; hidden: numLayers × batchSize × hiddenSize
        output = F.log_softmax(self.out(output), dim=2) # output: batchSize × 1 × outputSize
        return output, hidden, attentionWeight


#定义LuongAttention
class LuongAttention(nn.Module):
    #初始化
    def __init__(self, method, hiddenSize):
        super(LuongAttention, self).__init__()
        self.method = method
        #三种模式，dot,general,concat
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.Wa = nn.Linear(hiddenSize, hiddenSize)
        elif self.method == 'concat':
            self.Wa = nn.Linear(hiddenSize*2, hiddenSize)
            self.v = nn.Parameter(torch.FloatTensor(1, hiddenSize)) # self.v: 1 × hiddenSize

    #给出dot计算方法
    def dot_score(self, hidden, encoderOutput):

        return torch.sum(hidden*encoderOutput, dim=2)

    # 给出general计算方法
    def general_score(self, hidden, encoderOutput):

        energy = self.Wa(encoderOutput) # energy: batchSize × seq_len × hiddenSize
        return torch.sum(hidden*energy, dim=2)

    # 给出gconcat计算方法
    def concat_score(self, hidden, encoderOutput):
        # hidden: batchSize × 1 × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        energy = torch.tanh(self.Wa(torch.cat((hidden.expand(-1, encoderOutput.size(1), -1), encoderOutput), dim=2))) # energy: batchSize × seq_len × hiddenSize
        return torch.sum(self.v*energy, dim=2)

    # 定义前向计算
    def forward(self, hidden, encoderOutput):
        #确定使用哪种计算方式，3选1
        if self.method == 'general':
            attentionScore = self.general_score(hidden, encoderOutput)
        elif self.method == 'concat':
            attentionScore = self.concat_score(hidden, encoderOutput)
        elif self.method == 'dot':
            attentionScore = self.dot_score(hidden, encoderOutput)
        # attentionScore: batchSize × seq_len
        return F.softmax(attentionScore, dim=1).unsqueeze(1) # => batchSize × 1 × seq_len


 # 定义LuongAttentionDecoder
class LuongAttentionDecoderRNN(nn.Module):
    #初始化
    def __init__(self, featureSize, hiddenSize, outputSize, embedding, numLayers=1, dropout=0.1, attnMethod='dot'):
        super(LuongAttentionDecoderRNN, self).__init__()
        #对输入进行dropout
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        #核心api，搭建GRU
        self.gru = nn.GRU(featureSize, hiddenSize, num_layers=numLayers, dropout=(0 if numLayers==1 else dropout), batch_first=True)
        #定义权重计算和联合方式
        self.attention_weight = LuongAttention(attnMethod, hiddenSize)
        self.attention_combine = nn.Linear(hiddenSize*2, hiddenSize)
        self.out = nn.Linear(hiddenSize, outputSize)
        self.numLayers = numLayers

    # 定义前向计算
    def forward(self, inputStep, hidden, encoderOutput):
        # inputStep: batchSize × 1; hidden: numLayers × batchSize × hiddenSize
        #对输入做dropout
        inputStep = self.embedding(inputStep) # => batchSize × 1 × feaSize
        inputStep = self.dropout(inputStep)
        output, hidden = self.gru(inputStep, hidden) # output: batchSize × 1 × hiddenSize; hidden: numLayers × batchSize × hiddenSize
        attentionWeight = self.attention_weight(output, encoderOutput) # batchSize × 1 × seq_len
        # encoderOutput: batchSize × seq_len × hiddenSize
        context = torch.bmm(attentionWeight, encoderOutput) # context: batchSize × 1 × hiddenSize
        attentionCombine = self.attention_combine(torch.cat((output, context), dim=2)) # attentionCombine: batchSize × 1 × hiddenSize
        attentionOutput = torch.tanh(attentionCombine) # attentionOutput: batchSize × 1 × hiddenSize
        output = F.log_softmax(self.out(attentionOutput), dim=2) # output: batchSize × 1 × outputSize
        return output, hidden, attentionWeight

# 如何去选择decoder L or B or DecodeRnn
#定义Decoder
def _DecoderRNN(attnType, featureSize, hiddenSize, outputSize, embedding, numLayers, dropout, attnMethod):
    '''
    :param attnType: 选择L or B这两种decoder
    :param featureSize:特征的维度
    :param hiddenSize:隐层的大小
    :param outputSize:输出得大小
    :param embedding:embedding
    :param numLayers:内部几层
    :param dropout:正则去除过拟合
    :param attnMethod:dot,general,concat
    :return:
    '''
    #使用哪种attention
    if attnType not in ['L', 'B', None]:
        raise ValueError(attnType, "is not an appropriate attention type.")
    if attnType == 'L':
        return LuongAttentionDecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=numLayers, dropout=dropout, attnMethod=attnMethod)
    elif attnType == 'B':
        return BahdanauAttentionDecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=numLayers, dropout=dropout)
    else:
        return DecoderRNN(featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=numLayers, dropout=dropout)


#定义核心类seq2seq
class Seq2Seq:
    #初始化

    def __init__(self, dataClass, featureSize, hiddenSize, encoderNumLayers=1, decoderNumLayers=1, attnType='L', attnMethod='dot', dropout=0.1, encoderBidirectional=False, outputSize=None, embedding=None, device=torch.device("cpu")):
        # dataclass 是数据
        outputSize = outputSize if outputSize else dataClass.wordNum
        # 对话的长度，数据的长度。
        embedding = embedding if embedding else nn.Embedding(outputSize+1, featureSize)
        #数据读入
        self.dataClass = dataClass
        #搭建模型架构的内容。
        self.featureSize, self.hiddenSize = featureSize, hiddenSize
        # 准备encoder  调用构建
        self.encoderRNN = EncoderRNN(featureSize, hiddenSize, embedding=embedding, numLayers=encoderNumLayers, dropout=dropout, bidirectional=encoderBidirectional).to(device)
        # 准备decoder  调用构建
        self.decoderRNN = _DecoderRNN(attnType, featureSize, hiddenSize, outputSize, embedding=embedding, numLayers=decoderNumLayers, dropout=dropout, attnMethod=attnMethod).to(device)
        self.embedding = embedding.to(device)
        self.device = device

        #定义训练方法
    def train(self, batchSize, isDataEnhance=False, dataEnhanceRatio=0.2, epoch=100, stopRound=10, lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, teacherForcingRatio=0.5):
        # 使用那个api训练
        self.encoderRNN.train(), self.decoderRNN.train()
        #给定batchSize 和 是否数据增广isDataEnhance
        batchSize = min(batchSize, self.dataClass.trainSampleNum) if batchSize>0 else self.dataClass.trainSampleNum
        dataStream = self.dataClass.random_batch_data_stream(batchSize=batchSize, isDataEnhance=isDataEnhance, dataEnhanceRatio=dataEnhanceRatio)
        #定义优化器，使用adam
        # 对于测试数据batch制作。testsize默认0.2 ，即为0.2比例的测试数据。
        if self.dataClass.testSize>0: testStrem = self.dataClass.random_batch_data_stream(batchSize=batchSize, type='test')
        itersPerEpoch = self.dataClass.trainSampleNum//batchSize # 训练数据总长度//批次大小  = preEpoch
        # 定义有优化器
        encoderOptimzer = torch.optim.Adam(self.encoderRNN.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        decoderOptimzer = torch.optim.Adam(self.decoderRNN.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # 记忆个时间
        st = time.time()
        #做每个epoch循环
        for e in range(epoch):
            for i in range(itersPerEpoch):
                X, XLens, Y, YLens = next(dataStream)
                loss = self._train_step(X, XLens, Y, YLens, encoderOptimzer, decoderOptimzer, teacherForcingRatio)
                #计算bleu的参考指标
                if (e*itersPerEpoch+i+1)%stopRound==0:
                    bleu = _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                    embAve = _embAve_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                    print("After iters %d: loss = %.3lf; train bleu: %.3lf, embAve: %.3lf; "%(e*itersPerEpoch+i+1, loss, bleu, embAve), end='')
                    if self.dataClass.testSize>0:
                        X, XLens, Y, YLens = next(testStrem)
                        bleu = _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                        embAve = _embAve_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, self.dataClass.maxSentLen, device=self.device)
                        print('test bleu: %.3lf, embAve: %.3lf; '%(bleu, embAve), end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*batchSize
                    speed = (e*itersPerEpoch+i+1)*batchSize/(time.time()-st)
                    print("%.3lf qa/s; remaining time: %.3lfs;"%(speed, restNum/speed))
    #保存model
    def save(self, path):
        torch.save({"encoder":self.encoderRNN, "decoder":self.decoderRNN, 
                    "word2id":self.dataClass.word2id, "id2word":self.dataClass.id2word}, path)
        print('Model saved in "%s".'%path)
  #训练中的梯度及loss计算 bp
    def _train_step(self, X, XLens, Y, YLens, encoderOptimzer, decoderOptimzer, teacherForcingRatio):
        #计算梯度 BP  模型梯度初始化为0
        encoderOptimzer.zero_grad()
        decoderOptimzer.zero_grad()
        #计算loss
        loss, nTotal = _calculate_loss(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, teacherForcingRatio, device=self.device)
        #实现反向传播 loss/nTotal 平均生成一个字符的产生的loss，本身返回的loss是生成一句话，即连续decoder的loss和。
        (loss/nTotal).backward()
        # 更新所有的参数
        encoderOptimzer.step()
        decoderOptimzer.step()

        return loss.item() / nTotal

#读入预处理的数据进行操作
# 在train.py中运行记得打开
from model.pre_process import seq2id, id2seq, filter_sent


class ChatBot:
    def __init__(self, modelPath, device=torch.device('cpu')):   #初始化
        # 从模型中加载网络结构，和词字典。
        modelDict = torch.load(modelPath)
        self.encoderRNN, self.decoderRNN = modelDict['encoder'].to(device), modelDict['decoder'].to(device)
        self.word2id, self.id2word = modelDict['word2id'], modelDict['id2word']
        self.hiddenSize = self.encoderRNN.hiddenSize
        self.device = device
        # 当前模型是在验证
        self.encoderRNN.eval(), self.decoderRNN.eval()

    #定义贪婪搜索，inference时使用
    def predictByGreedySearch(self, inputSeq, maxAnswerLength=32, showAttention=False, figsize=(12,6)):
        # inputseq ：输入的问句
        inputSeq = filter_sent(inputSeq) # 去掉停用词
        inputSeq = [w for w in jieba.lcut(inputSeq) if w in self.word2id.keys()]  #先做分词
        X = seq2id(self.word2id, inputSeq)  # 变成单词的编号
        XLens = torch.tensor([len(X)+1], dtype=torch.int, device=self.device)   #处理输入成张量，计算长度，加结束符
        X = X + [eosToken] # 加终止符
        X = torch.tensor([X], dtype=torch.long, device=self.device)  # x输入数据转化成张量
        #定义相关的层，确定相应的encoder，确定隐层
        d = int(self.encoderRNN.bidirectional)+1 # 如果是双向的需要numlayers *2
        hidden = torch.zeros((d*self.encoderRNN.numLayers, 1, self.hiddenSize), dtype=torch.float32, device=self.device)
        encoderOutput, hidden = self.encoderRNN(X, XLens, hidden)
        hidden = hidden[-d*self.decoderRNN.numLayers::2].contiguous()  # 这个列表中的：：2 ，猜测可能是 d 这个变量。也没有报错，因为默认的是BI 的模式

        attentionArrs = []
        Y = []
        decoderInput = torch.tensor([[sosToken]], dtype=torch.long, device=self.device)  #给定decoder的输入 sos起始字符
        while decoderInput.item() != eosToken and len(Y)<maxAnswerLength:   #确定输出的序列，同时使用attention计算权重，选取最优解
            decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
            # top v: seq中分值最大的 value ，i 代表index
            topv, topi = decoderOutput.topk(1)  # 排序h ttps://www.jb51.net/article/177713.htm
            decoderInput = topi[:,:,0]  # 跟新decoderinput
            attentionArrs.append(decoderAttentionWeight.data.cpu().numpy().reshape(1,XLens))
            Y.append(decoderInput.item())
        outputSeq = id2seq(self.id2word, Y)  # id 序列在词典中配对成文字序列
        if showAttention:   #  是否可视化attention，
            attentionArrs = np.vstack(attentionArrs)  # 竖直(按列顺序)把数组给堆叠起来
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot('111')
            cax = ax.matshow(attentionArrs, cmap='bone')
            fig.colorbar(cax)
            ax.set_xticklabels(['', '<SOS>'] + inputSeq)
            ax.set_yticklabels([''] + outputSeq)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            plt.show()
        return ''.join(outputSeq[:-1])

#beamsearch的定义，inference时使用，计算量比贪婪算法大 基束搜索
    def predictByBeamSearch(self, inputSeq, beamWidth=10, maxAnswerLength=32, alpha=0.7, isRandomChoose=False, allRandomChoose=False, improve=True, showInfo=False):
        # 定义输出
        outputSize = len(self.id2word)
        inputSeq = filter_sent(inputSeq) #去停用词
        inputSeq = [w for w in jieba.lcut(inputSeq) if w in self.word2id.keys()]#分词

        X = seq2id(self.word2id, inputSeq)
        XLens = torch.tensor([len(X)+1], dtype=torch.int, device=self.device) #输入转tensor 同时加结束符
        X = X + [eosToken]
        X = torch.tensor([X], dtype=torch.long, device=self.device)

        #  默认参数情况下，使用双向gru encoder 和2层GRU decoder
        d = int(self.encoderRNN.bidirectional)+1
        # 初始化hidden
        hidden = torch.zeros((d*self.encoderRNN.numLayers, 1, self.hiddenSize), dtype=torch.float32, device=self.device)

        encoderOutput, hidden = self.encoderRNN(X, XLens, hidden)
        hidden = hidden[-d*self.decoderRNN.numLayers::2].contiguous()

        #把搜索宽度和最大回答长度做个数组 beamWidth行maxAnswerLength列
        Y = np.ones([beamWidth, maxAnswerLength], dtype='int32')*eosToken
        # prob: beamWidth × 1 下限的概率0  beamWidth行每一行seq的打分
        prob = np.zeros([beamWidth, 1], dtype='float32')
        #初始化输出seq 一个起始符开始。
        decoderInput = torch.tensor([[sosToken]], dtype=torch.long, device=self.device)
        # decoderOutput: 1 × 1 × outputSize; hidden: numLayers × 1 × hiddenSize 
        decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
        # topv: 1 × 1 × beamWidth; topi: 1 × 1 × beamWidth
        # 在贪心算法中只取了topk(1)，这里beamWidth。
        topv, topi = decoderOutput.topk(beamWidth)
        # decoderInput: beamWidth × 1
        decoderInput = topi.view(beamWidth, 1)
        for i in range(beamWidth):
            Y[i, 0] = decoderInput[i].item()
        Y_ = Y.copy() # 将第一个字符id 的beamWidth个结果 放在第一列的位置（每一行，0列）
        prob += topv.view(beamWidth, 1).data.cpu().numpy()
        prob_ = prob.copy()  # 第一个字符id beamWidth个结果的概率

        # hidden: numLayers × beamWidth × hiddenSize
        hidden = hidden.expand(-1, beamWidth, -1).contiguous()
        localRestId = np.array([i for i in range(beamWidth)], dtype='int32')
        encoderOutput = encoderOutput.expand(beamWidth, -1, -1) # => beamWidth × 1 × hiddenSize
        for i in range(1, maxAnswerLength):# 从第二个词开始，每个词进行计算

            # decoderOutput: beamWidth × 1 × outputSize; hidden: numLayers × beamWidth × hiddenSize; decoderAttentionWeight: beamWidth × 1 × XSeqLen
            decoderOutput, hidden, decoderAttentionWeight = self.decoderRNN(decoderInput, hidden, encoderOutput)
            # topv: beamWidth × 1; topi: beamWidth × 1
            if improve:
                decoderOutput = decoderOutput.view(-1, 1)
                if allRandomChoose:
                    topv, topi = self._random_pick_k_by_prob(decoderOutput, k=beamWidth)
                else:
                    topv, topi = decoderOutput.topk(beamWidth, dim=0)
            else:
                topv, topi = (torch.tensor(prob[localRestId], dtype=torch.float32, device=self.device).unsqueeze(2)+decoderOutput).view(-1,1).topk(beamWidth, dim=0)
            # decoderInput: beamWidth × 1
            decoderInput = topi%outputSize
            #计算过程，主要算概率，算路径上的最大概率
            idFrom = topi.cpu().view(-1).numpy()//outputSize
            Y[localRestId, :i+1] = np.hstack([Y[localRestId[idFrom], :i], decoderInput.cpu().numpy()])
            prob[localRestId] = prob[localRestId[idFrom]] + topv.data.cpu().numpy()
            hidden = hidden[:, idFrom, :]

            restId = (decoderInput!=eosToken).cpu().view(-1)
            localRestId = localRestId[restId.numpy().astype('bool')]
            decoderInput = decoderInput[restId]
            hidden = hidden[:, restId, :]
            encoderOutput = encoderOutput[restId]
            beamWidth = len(localRestId)
            if beamWidth<1:  #直到搜索宽度为0
                break
        lens = [i.index(eosToken) if eosToken in i else maxAnswerLength for i in Y.tolist()]
        ans = [''.join(id2seq(self.id2word, i[:l])) for i,l in zip(Y,lens)]
        prob = [prob[i,0]/np.power(lens[i], alpha) for i in range(len(ans))]
        # 上边的代码主要对结果，长度，概率，做了个定义和计算

        # 给定结果id的策略
        if isRandomChoose or allRandomChoose:    #对于回答方面做的策略，会去prob最大的那个，同时也可以给出概率
            prob = [np.exp(p) for p in prob]
            prob = [p/sum(prob) for p in prob]
            if showInfo:
                for i in range(len(ans)):
                    print((ans[i], prob[i]))
            return random_pick(ans, prob)
        else:
            ansAndProb = list(zip(ans,prob))
            ansAndProb.sort(key=lambda x: x[1], reverse=True)
            if showInfo:
                for i in ansAndProb:
                    print(i)
            return ansAndProb[0][0]

#定义验证方法
    def evaluate(self, dataClass, batchSize=128, isDataEnhance=False, dataEnhanceRatio=0.2, streamType='train'):
        dataClass.reset_word_id_map(self.id2word, self.word2id) #给定输入，同时初始化bleu等评价指标
        dataStream = dataClass.one_epoch_data_stream(batchSize=batchSize, isDataEnhance=isDataEnhance, dataEnhanceRatio=dataEnhanceRatio, type=streamType)
        bleuScore, embAveScore = 0.0, 0.0
        totalSamplesNum = dataClass.trainSampleNum if streamType=='train' else dataClass.testSampleNum#选用test数据
        iters = 0
        st = time.time()
        while True:   #验证的循环中主要完成计算bleu和embave的评分同时打印出来
            try:
                X, XLens, Y, YLens = next(dataStream)
            except:
                break
            bleuScore += _bleu_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, dataClass.maxSentLen, self.device, mean=False)
            embAveScore += _embAve_score(self.encoderRNN, self.decoderRNN, X, XLens, Y, YLens, dataClass.maxSentLen, self.device, mean=False)
            iters += len(X)
            finishedRatio = iters/totalSamplesNum
            # 打印时间，完成比例
            print('Finished %.3lf%%; remaining time: %.3lfs'%(finishedRatio*100.0, (time.time()-st)*(1.0-finishedRatio)/finishedRatio))
        return bleuScore/totalSamplesNum, embAveScore/totalSamplesNum

    def _random_pick_k_by_prob(self, decoderOutput, k):#根据概率随机取K个结果
        # decoderOutput: beamWidth*outputSize × 1
        df = pd.DataFrame([[i] for i in range(len(decoderOutput))])
        prob = torch.softmax(decoderOutput.data, dim=0).cpu().numpy().reshape(-1)
        topi = torch.tensor(np.array(df.sample(n=k, weights=prob)), dtype=torch.long, device=self.device)
        return decoderOutput[topi.view(-1)], topi


def random_pick(sample, prob):#随机pick一个prob比较大的
    x = random.uniform(0,1)
    cntProb = 0.0
    for sampleItem, probItem in zip(sample, prob):
        cntProb += probItem
        if x < cntProb:break
    return sampleItem
#bleu的评价指标，机器翻译的指标，最起码句子能比较顺，能读的通
def _bleu_score(encoderRNN, decoderRNN, X, XLens, Y, YLens, maxSentLen, device, mean=True):
    Y_pre = _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, maxSentLen, teacherForcingRatio=0, device=device)
    Y = [list(Y[i])[:YLens[i]-1] for i in range(len(YLens))]
    Y_pre = Y_pre.cpu().data.numpy()
    Y_preLens = [list(i).index(0) if 0 in i else len(i) for i in Y_pre]
    Y_pre = [list(Y_pre[i])[:Y_preLens[i]] for i in range(len(Y_preLens))]
    bleuScore = [sentence_bleu([i], j, weights=(1,0,0,0)) for i,j in zip(Y, Y_pre)]
    return np.mean(bleuScore) if mean else np.sum(bleuScore)
#embAve的评价指标，类似平方差之类的
def _embAve_score(encoderRNN, decoderRNN, X, XLens, Y, YLens, maxSentLen, device, mean=True):
    Y_pre = _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, maxSentLen, teacherForcingRatio=0, device=device)
    Y_pre = Y_pre.data
    Y_preLens = [list(i).index(0) if 0 in i else len(i) for i in Y_pre]
    # emb代表embedding
    emb = encoderRNN.embedding
    # 每个单独的字符变成词嵌入向量
    Y, Y_pre = emb(torch.tensor(Y, dtype=torch.long, device=device)).cpu().data.numpy(), emb(Y_pre).cpu().data.numpy()
    # 每一句话的中的词的嵌入向量均值
    sentVec = np.array([np.mean(Y[i,:YLens[i]], axis=0) for i in range(len(Y))], dtype='float32')
    sent_preVec = np.array([np.mean(Y_pre[i,:Y_preLens[i]], axis=0) for i in range(len(Y_pre))], dtype='float32')
    # 计算Ave
    embAveScore = np.sum(sentVec*sent_preVec, axis=1)/(np.sqrt(np.sum(np.square(sentVec), axis=1))*np.sqrt(np.sum(np.square(sent_preVec), axis=1)))
    return np.mean(embAveScore) if mean else np.sum(embAveScore)
   #计算loss
def _calculate_loss(encoderRNN, decoderRNN, X, XLens, Y, YLens, teacherForcingRatio, device):
    featureSize, hiddenSize = encoderRNN.featureSize, encoderRNN.hiddenSize
    # X: batchSize × XSeqLen; Y: batchSize × YSeqLen
    X, Y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)#转tensor
    XLens, YLens = torch.tensor(XLens, dtype=torch.int, device=device), torch.tensor(YLens, dtype=torch.int, device=device)

    batchSize = X.size(0)
    XSeqLen, YSeqLen = X.size(1), YLens.max().item()
    # 初始化encoderOutput
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype=torch.float32, device=device)

    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers, batchSize, hiddenSize), dtype=torch.float32, device=device)
    # XLens进行降序排序
    XLens, indices = torch.sort(XLens, descending=True)
    #索引回归升序排序=desortedIndices
    _, desortedIndices = torch.sort(indices, descending=False)
    encoderOutput, hidden = encoderRNN(X[indices], XLens, hidden)
    # encoderOutput按desortedIndices 重排 对应原始的 X
    encoderOutput, hidden = encoderOutput[desortedIndices], hidden[-d*decoderRNN.numLayers::d, desortedIndices, :] #hidden[:decoderRNN.numLayers, desortedIndices, :]
    # 初始化decoderInput seq,以一个开始字符开始。
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long, device=device)
    loss, nTotal = 0, 0
    for i in range(YSeqLen):
        #遍历  对于每个decoder的中，都会取top，并计算loss，训练过程中对比训练数据和真实数据之间的差
        # decoderOutput: batchSize × 1 × outputSize
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)
        loss += F.nll_loss(decoderOutput[:,0,:], Y[:,i], reduction='sum')
        nTotal += len(decoderInput)
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:,i:i+1]
        else:
            topv, topi = decoderOutput.topk(1)
            decoderInput = topi[:,:,0]# topi.squeeze().detach()
        # 过滤掉长度已经小于当次循环长度为i的句子。
        restId = (YLens>i+1).view(-1)
        decoderInput = decoderInput[restId]
        hidden = hidden[:, restId, :]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        YLens = YLens[restId]
    # loss,和循环的计数。
    return loss, nTotal
#计算Y的预测值
def _calculate_Y_pre(encoderRNN, decoderRNN, X, XLens, Y, YMaxLen, teacherForcingRatio, device):
    featureSize, hiddenSize = encoderRNN.featureSize, encoderRNN.hiddenSize
    # X: batchSize × XSeqLen; Y: batchSize × YSeqLen
    X, Y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(Y, dtype=torch.long, device=device)  #给定输入
    XLens = torch.tensor(XLens, dtype=torch.int, device=device)

    batchSize = X.size(0)
    XSeqLen = X.size(1)
    encoderOutput = torch.zeros((batchSize, XSeqLen, featureSize), dtype=torch.float32, device=device)  #encoder输出

    d = int(encoderRNN.bidirectional)+1
    hidden = torch.zeros((d*encoderRNN.numLayers, batchSize, hiddenSize), dtype=torch.float32, device=device)
    # # XLens进行降序排序
    XLens, indices = torch.sort(XLens, descending=True)
    _, desortedIndices = torch.sort(indices, descending=False)  #排序
    encoderOutput, hidden = encoderRNN(X[indices], XLens, hidden)
    encoderOutput, hidden = encoderOutput[desortedIndices], hidden[-d*decoderRNN.numLayers::d, desortedIndices, :] #hidden[:decoderRNN.numLayers, desortedIndices, :]
    decoderInput = torch.tensor([[sosToken] for i in range(batchSize)], dtype=torch.long, device=device)#把encoder的输出接入到decoder输入中
    Y_pre, localRestId = torch.ones([batchSize, YMaxLen], dtype=torch.long, device=device)*eosToken, torch.tensor([i for i in range(batchSize)], dtype=torch.long, device=device)
    for i in range(YMaxLen):  #循环 把每一个batch中的y_pre的得到（使用attention的权重）
        # decoderOutput: batchSize × 1 × outputSize
        decoderOutput, hidden, decoderAttentionWeight = decoderRNN(decoderInput, hidden, encoderOutput)
        if random.random() < teacherForcingRatio:
            decoderInput = Y[:,i:i+1]
        else:
            topv, topi = decoderOutput.topk(1)#取top1
            decoderInput = topi[:,:,0]# topi.squeeze().detach()
        Y_pre[localRestId, i] = decoderInput.squeeze()
        restId = (decoderInput!=eosToken).view(-1)
        localRestId = localRestId[restId]
        decoderInput = decoderInput[restId]
        hidden = hidden[:, restId, :]
        encoderOutput = encoderOutput[restId]
        Y = Y[restId]
        if len(localRestId)<1:
            break
    return Y_pre


if __name__ == '__main__':
    # 读入数据
    dataClass = Corpus('../data/qingyun.tsv', maxSentenceWordsNum=25)

    # 指定模型和一些超参
    model = Seq2Seq(dataClass, featureSize=128, hiddenSize=128,
                    attnType='L', attnMethod='concat',
                    encoderNumLayers=5, decoderNumLayers=3,
                    encoderBidirectional=True,
                    device=torch.device('cuda:0'))
    model.train(batchSize=512, epoch=1000)