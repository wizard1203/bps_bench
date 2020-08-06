import json
import os
import xlwt

file_name = 'traces1'

dir_path = './%s/' % file_name
dir_list = os.listdir(dir_path)
f_store = open('./bps_logs/%s.txt' % file_name,'w')
f_excel = xlwt.Workbook(encoding='utf-8')
sheet1 = f_excel.add_sheet('Trace')
row0 = ['configuration','avg_BROADCAST','avg_COPYH2D','acg_COPYD2H','avg_REDUCE','avg_PULL','avg_PUSH']

class AverageMeter(object):
    """Computes and stores the average and currentcurrent value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



for i in range(0,len(row0)):
	sheet1.write(0,i,row0[i])
row = 1
for j in range(len(dir_list)):
	dir_path_all = dir_path + (dir_list[j]+'/3/')
	print(dir_path_all)




	with open(dir_path_all+"comm.json",'r') as load_f:
	  load_dict = json.load(load_f)
	temp = load_dict['traceEvents']

	BROADCAST = []
	COPYH2D = []
	COPYD2H = []
	REDUCE = []
	PULL = []
	PUSH = []
	for i in range(len(temp)):
		# print(temp[i])
		json_temp = json.loads(json.dumps(temp[i]))
		if json_temp['name'] == "Comm.byteps.Whole.Gradient.BROADCAST":
			BROADCAST.append(json_temp['dur'])
		elif json_temp['name'] == "Comm.byteps.Whole.Gradient.COPYH2D":
			COPYH2D.append(json_temp['dur'])
		elif json_temp['name'] == "Comm.byteps.Whole.Gradient.COPYD2H":
			COPYD2H.append(json_temp['dur'])
		elif json_temp['name'] == "Comm.byteps.Whole.Gradient.REDUCE":
			REDUCE.append(json_temp['dur'])
		elif json_temp['name'] == "Comm.byteps.Whole.Gradient.PUSH":
			PUSH.append(json_temp['dur'])
		elif json_temp['name'] == "Comm.byteps.Whole.Gradient.PULL":
			PULL.append(json_temp['dur'])
	print(len(BROADCAST))
	print(len(COPYH2D))
	print(len(COPYD2H))
	print(len(REDUCE))
	print(len(PULL))
	print(len(PUSH))
	sum_BROADCAST = 0
	sum_COPYH2D = 0
	sum_COPYD2H = 0
	sum_REDUCE = 0
	sum_PULL = 0
	sum_PUSH = 0
	for i in range(len(BROADCAST)):
		sum_BROADCAST += BROADCAST[i]
	for i in range(len(COPYH2D)):
		sum_COPYH2D += COPYH2D[i]
	for i in range(len(COPYD2H)):
		sum_COPYD2H += COPYD2H[i]
	for i in range(len(REDUCE)):
		sum_REDUCE += REDUCE[i]
	for i in range(len(PULL)):
		sum_PULL += PULL[i]
	for i in range(len(PUSH)):
		sum_PUSH += PUSH[i]

	f_store.write(dir_list[j]+'\n')
	sheet1.write(row,0,dir_list[j])
	if len(BROADCAST) != 0:
		print('avg BROADCAST',sum_BROADCAST/len(BROADCAST))
		f_store.write('avg BROADCAST: '+str(sum_BROADCAST/len(BROADCAST))+'\n')
		sheet1.write(row,1,int(sum_BROADCAST/len(BROADCAST)))
	if len(COPYH2D) != 0:
		print('avg COPYH2D',sum_COPYH2D/len(COPYH2D))
		f_store.write('avg COPYH2D: '+str(sum_COPYH2D/len(COPYH2D))+'\n')
		sheet1.write(row,2,int(sum_COPYH2D/len(COPYH2D)))
	if len(COPYD2H) != 0:
		print('avg COPYD2H',sum_COPYD2H/len(COPYD2H))
		f_store.write('avg COPYD2H: '+str(sum_COPYD2H/len(COPYD2H))+'\n')
		sheet1.write(row,3,int(sum_COPYD2H/len(COPYD2H)))
	if len(REDUCE) != 0:
		print('avg REDUCE',sum_REDUCE/len(REDUCE))
		f_store.write('avg REDUCE: '+str(sum_REDUCE/len(REDUCE))+'\n')
		sheet1.write(row,4,int(sum_REDUCE/len(REDUCE)))
	if len(PULL) != 0:
		print('avg PULL',sum_PULL/len(PULL))
		f_store.write('avg PULL: '+str(sum_PULL/len(PULL))+'\n')
		sheet1.write(row,5,int(sum_PULL/len(PULL)))
	if len(PUSH) != 0:
		print('avg PUSH',sum_PUSH/len(PUSH))
		f_store.write('avg PUSH: '+str(sum_PUSH/len(PUSH))+'\n')
		sheet1.write(row,6,int(sum_PUSH/len(PUSH)))
	row += 1
	f_store.write('\n')


f_excel.save('./bps_logs/%s.xls' % file_name)