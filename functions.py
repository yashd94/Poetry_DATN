import modules.text_processing
import torch
import torch.nn.functional as F
from imp import reload
from modules.text_processing import read_split_file_lyrics, BatchWrapper, generate_iterators_lyrics

#Taken and modified from lab/hw
def evaluate_DATN(model, data_iter, batch_size, source_flag = False):
    model.eval()
    val_loss = 0
    total = 0
    total_acc = 0
    val_acc = 0
    for i in range(len(data_iter)):
        vectors,labels = next(data_iter.__iter__())
        if source_flag:
            attention, output = model(vectors)
        else:
            output = model(vectors)
        #_, predicted = torch.topk(output.data, k=2, dim=1)
        #predictions = torch.zeros(labels.size()).to("cuda")
        #predictions.scatter_(1, predicted, 1)
        val_acc += ((output > 0.5) == labels.byte()).sum().item()
        total_acc += labels.size(0)*2
        val_loss += F.kl_div(output.log(), labels)
        total +=1
    return val_loss / total, val_acc / total_acc

#Taken and modified from lab/hw
def training_loop_DATN(batch_size, num_epochs, model, 
                       loss_, optim, training_iter, 
                       dev_iter,source_flag = False, verbose=True):
    
    epoch = 0
    total_batches = int(len(training_iter))
    dev_accuracies = []
    test_accuracies = []
    while epoch <= num_epochs:
        print("Training...")
        for i in range(total_batches):
            model.train()
            vectors, labels = next(training_iter.__iter__())
            model.zero_grad()
            
            if source_flag:
            
                src_attention, output = model(vectors)
            
            else:
                
                output = model(vectors)
                
            lossy = loss_(output.log(), labels)
            lossy.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()


        model.eval()
        print("Evaluating dev...")
        eval_loss, eval_acc = evaluate_DATN(model, dev_iter, batch_size, source_flag = source_flag)
        
        dev_accuracies.append(eval_acc)
        if verbose:
            print("Epoch %i; Loss %f; Dev loss %f, Dev acc: %f"  %(epoch, lossy.item(), eval_loss, eval_acc))
        epoch += 1    
    best_dev = max(dev_accuracies)
    
    if source_flag:
        
        return best_dev, src_attention
    
    else:
        
        return best_dev
    
    