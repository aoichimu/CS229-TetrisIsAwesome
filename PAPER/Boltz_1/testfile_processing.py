import os
import numpy as np

#checkpoints_dir="./checkpoints"

#test_filename = "TESTFORTEST.txt"

def list_checkpoints(checkpoints_dir):
    """
    Retrieves the list of checkpoints available for testing
    together with their iteration times
    
    """
    iterations=[]
    paths=[]
    for filename in os.listdir(checkpoints_dir):
        if (filename.endswith(".meta")) and filename.startswith("dqn_breakout_RAM.ckpt-"):
            #print(os.path.join("../Testing_Algorithm/checkpoints", filename))
            chopped_name=filename.replace(".meta",""))
            T=np.int(chopped_name.replace("dqn_breakout_RAM.ckpt-",""))
            iterations.append(T)
            paths.append(chopped_name)
            continue
        else:
            continue

    sort_list=sorted(zip(iterations, paths))

    sorted_iterations=[x[0] for x in sort_list]
    sorted_paths=[x[1] for x in sort_list]

    return sorted_iterations, sorted_paths
    
def last_test(test_filename):
    """
    Finds the time corresponding to the last test to avoid overwritting good results
    
    """
    if os.path.exists(test_filename):
        f=open(test_filename,'r')
        lines=f.readlines()
        f.close()
        if len(lines) ==0:
            final_T= 0
        elif lines[-1]=="\n":
            if len(lines)>1:
                last_line=lines[-2]
                final_T=np.int(last_line.split(' ', 1)[0])
            else:
                final_T=0
        else:
            last_line=lines[-1]
            final_T=np.int(last_line.split(' ', 1)[0])
            
                
    else:
        print test_filename ,"does not exist"
        final_T=0
    return final_T
    #return -1
    
def get_testing_data(checkpoints_dir,test_filename):
    """
    Finds the checkpoint data that has not been utilized for testing*
    """
    iterations,paths=list_checkpoints(checkpoints_dir)
    final_T=last_test(test_filename)
    L=len(iterations)
    if L==0:
        return iterations, paths
    else:
        if final_T>iterations[-1]:
            return [], []
        else:
            A=next(i for i,v in enumerate(iterations) if v > final_T)
    
    return iterations[A:], paths[A:]
    
#checkpoints_dir="./checkpoints"
#test_filename = "TESTFORTEST.txt"  
#get_testing_data(checkpoints_dir,test_filename)
#list_checkpoints("./checkpoints")
