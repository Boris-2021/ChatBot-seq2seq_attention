
#导入库
from model.model import *
from model.pre_process import *
import torch

#载入模型
chatbot = ChatBot('modelA.pkl')

#针对问题1
print(chatbot.predictByGreedySearch("你好啊"))

#针对问题2
print(chatbot.predictByBeamSearch("什么是人工智能", isRandomChoose=True, beamWidth=10))
