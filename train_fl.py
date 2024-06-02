import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms
from torchvision.models import resnet18
import copy
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FLServer:
    def __init__(self) -> None:

        self.clients = []


    def add_client(self, fl_client):
        self.clients.append(fl_client)
        # print(f"Done adding new client with id: {fl_client.id}")
        pass

    def test(self, model):
        # print accuracy of model on each client
        test_metrics = []
        for flclient in self.clients:
            test_metric = flclient.test(model)
            test_metrics.append(test_metric)
            #flclient.test_loss.append(test_metric['loss'])
        #print(test_metrics)
    



class FLClient:

    def __init__(self, id, train_data, test_data):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.train_loss = []
        #self.com_round = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.test_loss = []
        self.num_epoch = 0
        self.train_loader = self.train_data #torch.utils.data.DataLoader(self.train_data, batch_size=512, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=512, shuffle=False)
        
    def training(self, model, num_epoch=2, lr = 0.01):
        
        self.num_epoch = num_epoch
        optimizer = optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
        
        model.train()
        for epoch in range(num_epoch):
            total_loss, correct, num_sample = 0, 0, 0
            for idb, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                loss = F.cross_entropy(output, target)
                total_loss += loss.item() * len(data)
                num_sample += len(data)
                
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss.backward()
                optimizer.step()
            print(f"At epoch {epoch} correct: {correct} / {num_sample} loss: {(total_loss/num_sample):.4f} acc: {(100. * correct/ num_sample):.4f}")

        total_loss /= num_sample
        accuracy = 100. * correct / num_sample
        
        # store accuracy and loss per com_round
        self.train_loss.append(total_loss)
        self.train_accuracy.append(accuracy)

        return model.state_dict(), {'loss': total_loss, 'accuracy': accuracy, '#data': num_sample, 'correct': correct}

    def test(self, model):

        model.eval()
        total_loss, correct, num_sample = 0, 0, 0
        for _, (data, target) in enumerate(self.test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * len(data)
            num_sample += len(data)
            
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss = total_loss/num_sample
        test_accuracy = (correct/num_sample) * 100
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)

        return {'id': self.id, 'loss': total_loss / num_sample, 'accuracy': 100. * correct / num_sample, '#data': num_sample, 'correct': correct}

    def plot(self,x,y,idc ,linestyle,line_width,num_client,color,label):
        # Plot accuracy per com_round of each client
        if num_client % 2 != 0:
            raise ValueError("Odd number of clients, cannot plot")
        else: 
            plt.subplot(2,num_client//2,idc + 1)
            plt.plot(x,y, linestyle = linestyle, linewidth = line_width,color = color,label = label)
            # plt.xlabel('Communication round')
            # plt.ylabel('Accuracy')
            # plt.title(f'client {idc}')
           


class Helper:
    def __init__(self) -> None:
        self.load_data()
        

    

    def load_data(self):
        # Load data from torch
        # List of dataset: MNIST, CIFAR-10, Fashion-MNIST
        # CIFAR-10 dataset
        transform = transforms.Compose([ transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) ) ])

        
        train_data = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, download=False, transform=transform)
        # first_image, first_label = train_data[0]
        # print("First image tensor:", first_image.shape)
        # print("Label of first image:", first_label)
        

        print(f'size of train data: {len(train_data)}, size of test data: {len(test_data)}')
        # print
        # # statistic about number of samples in each class
        train_stat = {}
        for data, label in train_data: # list: [[data, label], ...]
            if label not in train_stat:
                train_stat[label] = 0
            train_stat[label] += 1
        print(f"Training stat: {train_stat}")

        test_stat = {}

        for data, label in test_data: # list: [[data, label], ...]
            if label not in test_stat:
                test_stat[label] = 0
            test_stat[label] += 1
        print(f"Test stat: {test_stat}")

        # ds_classes = {}
        # for i, (_, label) in enumerate(test_data):
        #     ds_classes.setdefault(label, []).append(i)
        # print(f"ds_classes: {ds_classes}")
        #import IPython; IPython.embed()

        # train_data, test_data
        # FL: divided data -> N clients
        # non-IID, IID
        # IID: N client, K = 50.000 / N; [0..k-1], [k..2k-1], [2k..3k-1],.... [..49999]
        # Define FL settings
        
        num_clients = 10
        
        alpha = 0.5  # Dirichlet distribution concentration parameter

        def sample_dirichlet_train_data(train_dataset, no_participants, alpha=0.5, seed = 4):
            """
            Sample non-IID training data using Dirichlet distribution.

            Args:
            - train_dataset: The training dataset.
            - no_participants: Number of participants to split the data into.
            - alpha: Dirichlet concentration parameter.

            Returns:
            - A list of DataLoader objects, each corresponding to a participant.
            """
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                rng = np.random
    
            all_indices = np.arange(len(train_dataset))
            all_targets = np.array(train_dataset.targets)

            indices_per_class = {}
            for cls in np.unique(all_targets):
                indices_per_class[cls] = all_indices[all_targets == cls]

            participant_indices = [[] for _ in range(no_participants)]
            for cls, cls_indices in indices_per_class.items():
                rng.shuffle(cls_indices)
                proportions = rng.dirichlet([alpha] * no_participants)
                proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
                split_indices = np.split(cls_indices, proportions)
                for participant in range(no_participants):
                    participant_indices[participant].extend(split_indices[participant])

            data_loaders = []
            for participant in range(no_participants):
                data_subset = Subset(train_dataset, participant_indices[participant])
                data_loader = DataLoader(data_subset, batch_size=512, shuffle=True)
                data_loaders.append(data_loader)

            return data_loaders

        # Function to count the labels in a dataset
        def count_labels(dataset):
            labels = [dataset[i][1] for i in range(len(dataset))]
            return Counter(labels)

        # Split training data in a non-IID manner
        client_train_data = sample_dirichlet_train_data(train_data, num_clients, alpha,seed = 4)
        print(np.array(client_train_data[0]))
        # For the test data, we can still split it uniformly if desired
        client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)

        # Print the sizes of each label in each client's training data
        for i, loader in enumerate(client_train_data):
            label_counts = count_labels(loader.dataset)
            print(f"Client {i} label distribution: {label_counts}")

        # Print the sizes of each client's test data to verify
        for i, test_subset in enumerate(client_test_data):
            print(f"Client {i} has {len(test_subset)} test samples")

        # Split data in IID setting 
        # client_train_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
        # client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)
        
        
        # # import IPython; IPython.embed()
        # # stat data of each client
        # print(f'In train data:')
        # for idc in range(num_clients):
            
        #     data_client = client_train_loaders[idc]
        #     stat_client = {}
        #     for data, label in data_client: # list: [[data, label], ...]
        #         if label not in stat_client:
        #             stat_client[label] = 0
        #         stat_client[label] += 1
        #     print(f"Client number: {idc} with data: {stat_client}")
        
        # print(f'In test data:')
        # for idc in range(num_clients):
            
        #     data_client = client_test_data[idc]
        #     stat_client = {}
        #     for data, label in data_client: # list: [[data, label], ...]
        #         if label not in stat_client:
        #             stat_client[label] = 0
        #         stat_client[label] += 1
        #     print(f"Client number: {idc} with data: {stat_client}")
        
        # FL protocol: sever -> client: global model
        # for each epoch: client receives global model; training -> local modelepoch
        # client -> server: local model
        # sever: aggregation = Avg(weight num_client model); [ [1000 elements], .... [1000 elements]] -> [1000 elements]



              
        flsever = FLServer()
        for idc in range(num_clients):
            client = FLClient(idc, client_train_data[idc], client_test_data[idc])
            flsever.add_client(client)
        
        print('Training')
        # Training
        global_model = resnet18(weights=None, num_classes=10).to(device)

        num_com_round = 10
        for id_com_round in range(num_com_round):
            print(f"Training at communication round id: {id_com_round}")
            list_local_models = []

            global_weights = copy.deepcopy(global_model.state_dict())
            
            # Init zero parameters
            for key in global_weights.keys():
                global_weights[key] = torch.zeros_like(global_weights[key]).to(torch.float32)

           
            
            # First
         
            

           
            
            # First com_round do not random select
            if id_com_round == 0:
                for flclient in flsever.clients:
                    print(f"Start training client number: {flclient.id}")
                    client_weights, train_metrics = flclient.training(copy.deepcopy(global_model),num_epoch = 2, lr = 1.0)
                    print(f"Finish training client number: {flclient.id} with metrics: {train_metrics}")
                    # list_local_models.append(local_model_dict)
                    # import IPython; IPython.embed(); exit(0)s

                    # FedAvg
                    for _, key in enumerate(global_weights.keys()):
                        update_key = client_weights[key] / len(flsever.clients)
                        global_weights[key].add_(update_key)

                global_model.load_state_dict(global_weights)

            

                # Test
                flsever.test(global_model)
                
                print("**"*50)
                continue 
            
                
            # From round 1 -> ...: random select 5 in each round
             # Choose random 5 client among clients
            selected_clients = random.sample(flsever.clients, 5)
            
            non_selected_clients = [item for item in flsever.clients if item not in selected_clients]


            print(f'Selected clients in round {id_com_round}: {[client.id for client in selected_clients]}')
            print(f'Non-selected clients in round {id_com_round}: {[client.id for client in non_selected_clients]}')
            # selected train
            for flclient in selected_clients:
                print(f"Start training client number: {flclient.id}")
                client_weights, train_metrics = flclient.training(copy.deepcopy(global_model),num_epoch = 3, lr = 1.0)
                print(f"Finish training client number: {flclient.id} with metrics: {train_metrics}")
                # list_local_models.append(local_model_dict)
                # import IPython; IPython.embed(); exit(0)s

                # FedAvg
                for _, key in enumerate(global_weights.keys()):
                    update_key = client_weights[key] / len(selected_clients)
                    global_weights[key].add_(update_key)

            # non-selected not train
            for not_flclient in non_selected_clients:
                
                not_flclient.train_loss.append(not_flclient.train_loss[-1])
                not_flclient.train_accuracy.append(not_flclient.train_accuracy[-1])

            global_model.load_state_dict(global_weights)

            

            # Test
            flsever.test(global_model)
            
            print("**"*50)
            


      
            
            
            


        # Plot train vs test loss per com_round
        plt.figure(figsize = (18,10))
        plt.suptitle("Plot Train/Test Loss per Com_round")
        
        
        for client in flsever.clients:
            x = np.arange(1,num_com_round + 1)
            #print(f'len x = {len(x)}')
            y = np.array(client.train_loss)
            #print(f'len y = {len(y)}')
            z = np.array(client.test_loss)
            #print(f'len z = {len(z)}')
            print(f'TRAIN LOSS = {y}')
            print(f'TEST LOSS = {z}')
            client.plot(x,y,client.id,'-',0.5, num_clients, color = 'r', label = 'train')
            client.plot(x,z,client.id,'-',0.5, num_clients, color = 'b',label = 'test')
            plt.xlabel('Com_round')
            plt.ylabel('Loss')
            plt.title(f'client {client.id}')
            plt.legend(loc='upper right')


        plt.tight_layout()
        plt.savefig('clients_loss.png')
        plt.show()
        
        

        # Plot train vs test accuracy per com_round
        plt.figure(figsize = (18,10))
        plt.suptitle("Plot Train/Test Accuracy per Com_round")
        
        
        for client in flsever.clients:
            x = np.arange(1,num_com_round + 1)
            y = np.array(client.train_accuracy)
            z = np.array(client.test_accuracy)
            print(f'TEST ACCURACY = {z}')
            print(f'TRAIN ACCURACY = {y}')
            client.plot(x,y,client.id,'-',0.5, num_clients, color = 'r', label = 'train')
            client.plot(x,z,client.id,'-',0.5, num_clients, color = 'b',label = 'test')
            plt.xlabel('Com_round')
            plt.ylabel('Accuracy')
            plt.title(f'client {client.id}')
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('clients_accuracy.png')
        plt.show()
        
        
        
        
                


if __name__ == '__main__':
    # set fixed random seed
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # random.seed(random_seed)
    
    helper = Helper()
