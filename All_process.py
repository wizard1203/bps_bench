import os
import xlwt

str1 = './bps_logs/alexnet-bs32-iters16-nworkers1-nservers1id0.log'
dir_path = './bps_logs/'


def extract_excel(dir_path):

    f_store = open('result.txt','w')
    f_excel = xlwt.Workbook(encoding='utf-8')
    sheet1 = f_excel.add_sheet('Trace')
    row0 = ['configuration','avg_Dataload','avg_Forward','acg_BackWard','avg_Co_update']
    for i in range(0,len(row0)):
        sheet1.write(0,i,row0[i])
    row = 1

    dir1 = os.listdir(dir_path)
    for j in range(len(dir1)):
        # print(dir_name)
        dir_name = dir1[j]
        print(dir_name)
        f1 = open(dir_path+dir_name,'r')
        num = 0
        Com_update = []
        Dataload = []
        Forward = []
        Backward = []
        for line in f1:
            if 'Iteration 8' in line:
                num += 1
            if 'Iteration 15' in line:
                break
            if num == 4:
                if 'Dataload time' in line:
                    temp = line.split(':')
                    Dataload.append(float(temp[-1]))
                elif 'Forward time' in line:
                    temp = line.split(':')
                    Forward.append(float(temp[-1]))
                elif 'Backward time' in line:
                    temp = line.split(':')
                    Backward.append(float(temp[-1]))
                elif 'Communication and updating time' in line:
                    temp = line.split(':')
                    Com_update.append(float(temp[-1]))
        print(len(Dataload))
        print(len(Forward))
        print(len(Backward))
        print(len(Com_update))
        if len(Dataload)==0 or len(Forward)==0 or len(Backward)==0 or len(Com_update)==0:
            print("No information in this file")
            continue

        sheet1.write(row,0,dir_name)

        f_store.write(dir_name+'\n')
        sum = 0
        for i in range(len(Dataload)):
            sum += Dataload[i]
        print('avg_Dataload: ',sum/len(Dataload))
        f_store.write('avg_Dataload: '+str(sum/len(Dataload))+'\n')
        sheet1.write(row,1,int(sum*1e8/len(Dataload)))
        sum = 0
        for i in range(len(Forward)):
            sum += Forward[i]
        print('avg_Forward: ',sum/len(Forward))
        f_store.write('avg_Forward: '+str(sum/len(Forward))+'\n')
        sheet1.write(row,2,int(sum*1e8/len(Forward)))
        sum = 0
        for i in range(len(Backward)):
            sum += Backward[i]
        print('avg_Backward: ',sum/len(Backward))
        f_store.write('avg_Backward: '+str(sum/len(Backward))+'\n')
        sheet1.write(row,3,int(sum*1e8/len(Backward)))
        sum = 0
        for i in range(len(Com_update)):
            sum += Com_update[i]
        print('avg_Co_updata: ',sum/len(Com_update))
        f_store.write('avg_Co_updata: '+str(sum/len(Com_update))+'\n'+'\n')
        sheet1.write(row,4,int(sum*1e8/len(Com_update)))
        row += 1
    f1.close()
    f_store.close()
    f_excel.save('./bps.xls')

def extract_training_log(dir_path, model, DMLC_PS, batch_size, num_iters, nworkers, nservers, worker_id):
    file_name = str(model)+'-network'+str(DMLC_PS)+'-bs'+str(batch_size)+'-iters'+str(num_iters)+ \
        '-nworkers'+str(nworkers)+'-nservers'+str(nservers)+'id'+str(worker_id)+'.log'
    logfile = os.path.join(dir_path, file_name)
    f = open(logfile, 'r')

    for line in f.readlines():
        if line.find('Img/sec per GPU:') > 0:
            items = line.split('Img/sec per GPU:')
            mean = items[-1].strip().split()[0]
    f.close()
    return float(mean)



def extract_one_tensor_log(dir_path, tensor_size, KB, DMLC_PS, batch_size, num_iters, nworkers, nservers, worker_id, local_rank):
    # file_name = 'one_tensor_test_size'+str(tensor_size)+'-network'+str(DMLC_PS)+'-nworkers'+ \
    #     str(nworkers)+'-nservers'+str(nservers)+'worker'+str(worker_id)+'rank'+str(local_rank)+'.log'
    if KB == '1':
        file_name = 'one_tensor_test_size'+str(tensor_size)+'KB-network'+str(DMLC_PS)+'-nworkers'+ \
            str(nworkers)+'-nservers'+str(nservers)+'worker'+str(worker_id)+'rank'+str(local_rank)+'.log'
    else:
        file_name = 'one_tensor_test_size'+str(tensor_size)+'-network'+str(DMLC_PS)+'-nworkers'+ \
                str(nworkers)+'-nservers'+str(nservers)+'worker'+str(worker_id)+'rank'+str(local_rank)+'.log'
    logfile = os.path.join(dir_path, file_name)
    f = open(logfile, 'r')

    for line in f.readlines():
        if line.find('Iter time:') > 0:
            items = line.split('Iter time:')
            mean = items[-1].strip().split()[0]
    f.close()
    return float(mean)

def get_serialized_log(dir_path, training_or_tensor, model=[], tensor_size=[], KB='1', DMLC_PS=[], batch_size=[], 
    num_iters=[], nworkers=[], nservers=[], worker_id=[], local_rank=[]):

    data = []
    if training_or_tensor == 'training':
        for model_i in model:
            for DMLC_PS_i in DMLC_PS:
                for batch_size_i in batch_size:
                    for num_iters_i in num_iters:
                        for nworkers_i in nworkers:
                            for nservers_i in nservers:
                                for worker_id_i in worker_id:
                                    data.append(extract_training_log(dir_path, model_i, DMLC_PS_i, batch_size_i, num_iters_i,
                                        nworkers_i, nservers_i, worker_id_i))
    elif training_or_tensor == 'tensor':
        for tensor_size_i in tensor_size:
            for DMLC_PS_i in DMLC_PS:
                for batch_size_i in batch_size:
                    for num_iters_i in num_iters:
                        for nworkers_i in nworkers:
                            for nservers_i in nservers:
                                for worker_id_i in worker_id:
                                    for local_rank_i in local_rank:
                                        data.append(extract_one_tensor_log(dir_path, tensor_size_i, KB, DMLC_PS_i, batch_size_i, num_iters_i,
                                            nworkers_i, nservers_i, worker_id_i, local_rank_i))
    return data































