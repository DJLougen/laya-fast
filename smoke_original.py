import sys,json,time,torch
sys.path.insert(0,'source')
torch.set_num_threads(4)
from rl_agent_api import RLAgent
t=time.perf_counter(); agent=RLAgent('source',device='cpu')
print('CPU_LOAD_SECONDS',time.perf_counter()-t,flush=True)
qs={'route':{'type':'choice','instructions':'Which team should handle this request?','criteria':{'billing':'payments, invoices and refunds','technical':'software bugs and outages','sales':'new contracts and pricing'}},'refund':{'type':'noul','instructions':'Does the message explicitly request a refund?'}}
t=time.perf_counter(); res=agent.system_one('My card was charged twice. Please refund the duplicate charge.',qs)
print(json.dumps(res,indent=2)); print('COLD_CALL_SECONDS',time.perf_counter()-t)
