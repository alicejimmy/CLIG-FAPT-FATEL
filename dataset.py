import logging
import numpy as np
import math
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from augment.data_augmentention import CIFAR10_Augmentention, CIFAR100_Augmentention
from model.resnet18_model import resnet18_model
from algorithm.utils import accuracy

# CIFAR10
def cifar10_dataloader(batch_size, creation_method='CLIG', partial_rate=0.1, noisy_rate=0.2, data_ratio=1.0):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    # Download dataset
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    data, label = train_data.data, torch.Tensor(train_data.targets).long()
    # Generate candidate labelset
    labelset = generate_candidate_labelset(train_data=torch.tensor(data).permute(0, 3, 1, 2), train_labels=label, creation_method=creation_method, dataset_name='cifar10', partial_rate=partial_rate, noisy_rate=noisy_rate)

    # For the experiment of reduce the amount of data
    torch.manual_seed(2024)
    num_to_train = int(len(label) * data_ratio)
    random_idx = torch.randperm(len(label))[:num_to_train]
    selected_data = data[random_idx]
    selected_label = label[random_idx]
    selected_labelset = labelset[random_idx]

    # Create augmented samples and packaged dataset
    partial_matrix_dataset = CIFAR10_Augmentention(selected_data, selected_labelset.float(), selected_label.float())
    train_loader = DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    #### Calculate Jaccard similarity ####
    # Read cifar-10N data
    noise_label = torch.load('dataset_collection/CIFAR-10_human.pt')

    aggre_label = torch.from_numpy(noise_label['aggre_label'])
    random_label1 = torch.from_numpy(noise_label['random_label1'])
    random_label2 = torch.from_numpy(noise_label['random_label2'])
    random_label3 = torch.from_numpy(noise_label['random_label3'])
    worst_label = torch.from_numpy(noise_label['worse_label'])

    # Calculate Jaccard similarity
    num_data = label.shape[0]
    num_class = len(torch.unique(label))
    labelset_reality = torch.zeros(num_data, num_class)
    labelset_reality[torch.arange(num_data), aggre_label] = 1
    labelset_reality[torch.arange(num_data), random_label1] = 1
    labelset_reality[torch.arange(num_data), random_label2] = 1
    labelset_reality[torch.arange(num_data), random_label3] = 1
    labelset[torch.arange(num_data), worst_label] = 1

    jaccard_similarity = 0
    for i in range(num_data):
        set1 = set([i for i, val in enumerate(labelset[i]) if val == 1])
        set2 = set([i for i, val in enumerate(labelset_reality[i]) if val == 1])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_similarity += intersection / union if union != 0 else 0
    jaccard_similarity = jaccard_similarity/num_data
    logging.info('Jaccard similarity coefficient:{0}'.format(jaccard_similarity))
    
    return train_loader, test_loader

# CIFAR100_SM
def cifar100_small_mammals_dataloader(batch_size, creation_method='CLIG', partial_rate=0.1, noisy_rate=0.2):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    # Download dataset
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    df_form = pd.read_csv('dataset_collection/small_mammals_result.csv', header=None, index_col=0)

    # Find the index of 500 data in the form
    selected_files = df_form.loc['img'].tolist()
    df_form = df_form.drop(index=['img'])
    selected_files = [int(filename.split('.')[0]) for filename in selected_files]
    selected_files_idx = [0] * len(selected_files)

    # Get only the data of small_mammals
    train_img, train_label = [], []
    test_img, test_label = [], []
    selected_classes = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    label_map = {}
    for subclass in selected_classes:
        subclass_idx = train_data.class_to_idx[subclass]
        label_map[subclass_idx] = len(label_map)
        train_idx = [i for i, label in enumerate(train_data.targets) if label == subclass_idx]
        for i in train_idx:
            select_img = train_data.data[i]
            select_label = train_data.targets[i]
            train_img.append(select_img)
            train_label.append(select_label)
            if i in selected_files:
                selected_files_idx[selected_files.index(i)] = len(train_label)-1
        test_idx = [i for i, label in enumerate(test_data.targets) if label == subclass_idx]
        for i in test_idx:
            select_img = test_data.data[i]
            select_label = test_data.targets[i]
            test_img.append(select_img)
            test_label.append(select_label)
    
    # Re-encode label to 0-4
    train_label = [label_map[train_label] for train_label in train_label]
    test_label = [label_map[test_label] for test_label in test_label]
    df_form = df_form.apply(lambda x: x.map(lambda y: int(y) if str(y).isdigit() else y))
    df_form = df_form.replace(label_map)

    # Adjust data type and format
    data = np.array(train_img)
    label = torch.Tensor(train_label).long()
    test_img = torch.stack([test_transform(test_img[i]) for i in range(len(test_img))])
    test_label = torch.Tensor(test_label).long()
    test_data = TensorDataset(test_img, test_label)
    
    # Generate candidate labelset
    labelset = generate_candidate_labelset(train_data=torch.tensor(data).permute(0, 3, 1, 2), train_labels=label, creation_method=creation_method, dataset_name='cifar100_small_mammals', partial_rate=partial_rate, noisy_rate=noisy_rate)    
    
    # Create augmented samples and packaged dataset
    partial_matrix_dataset = CIFAR100_Augmentention(data, labelset.float(), label.float())
    train_loader = DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    #### Calculate Jaccard similarity ####
    selected_files_labelset = torch.index_select(labelset, 0, torch.tensor(selected_files_idx))
    selected_files_truelabels = torch.index_select(label, 0, torch.tensor(selected_files_idx))
    logging.info('The 500 images in the forms:')
    logging.info('Average candidate num:{0}'.format(selected_files_labelset.sum(1).mean()))
    logging.info('The proportion of true labels in candidate label sets:{0}'.format(((torch.where(selected_files_labelset[torch.arange(len(selected_files_truelabels)), selected_files_truelabels]==1, True, False)==True).sum())/len(selected_files_truelabels)))

    # Calculate Jaccard similarity
    num_data = selected_files_truelabels.shape[0]
    num_class = len(torch.unique(selected_files_truelabels))
    labelset_reality = torch.zeros(num_data, num_class)
    for i in range(len(selected_files_truelabels)):
        unique_answer = df_form[i+1].unique()
        for value in unique_answer:
            labelset_reality[i, value] = 1
    
    jaccard_similarity = 0
    for i in range(num_data):
        set1 = set([i for i, val in enumerate(selected_files_labelset[i]) if val == 1])
        set2 = set([i for i, val in enumerate(labelset_reality[i]) if val == 1])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_similarity += intersection / union if union != 0 else 0
    jaccard_similarity = jaccard_similarity/num_data
    logging.info('Jaccard similarity coefficient:{0}'.format(jaccard_similarity))

    return train_loader, test_loader

# CIFAR100_T
def cifar100_trees_dataloader(batch_size, creation_method='CLIG', partial_rate=0.1, noisy_rate=0.2):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    # Download dataset
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    df_form = pd.read_csv('dataset_collection/trees_result.csv', header=None, index_col=0)

    # Find the index of 500 data in the form
    selected_files = df_form.loc['img'].tolist()
    df_form = df_form.drop(index=['img'])
    selected_files = [int(filename.split('.')[0]) for filename in selected_files]
    selected_files_idx = [0] * len(selected_files)

    # Get only the data of trees
    train_img, train_label = [], []
    test_img, test_label = [], []
    selected_classes = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
    label_map = {}
    for subclass in selected_classes:
        subclass_idx = train_data.class_to_idx[subclass]
        label_map[subclass_idx] = len(label_map)
        train_idx = [i for i, label in enumerate(train_data.targets) if label == subclass_idx]
        for i in train_idx:
            select_img = train_data.data[i]
            select_label = train_data.targets[i]
            train_img.append(select_img)
            train_label.append(select_label)
            if i in selected_files:
                selected_files_idx[selected_files.index(i)] = len(train_label)-1
        test_idx = [i for i, label in enumerate(test_data.targets) if label == subclass_idx]
        for i in test_idx:
            select_img = test_data.data[i]
            select_label = test_data.targets[i]
            test_img.append(select_img)
            test_label.append(select_label)

    # Re-encode label to 0-4
    train_label = [label_map[train_label] for train_label in train_label]
    test_label = [label_map[test_label] for test_label in test_label]
    df_form = df_form.apply(lambda x: x.map(lambda y: int(y) if str(y).isdigit() else y))
    df_form = df_form.replace(label_map)
    
    # Adjust data type and format
    data = np.array(train_img)
    label = torch.Tensor(train_label).long()
    test_img = torch.stack([test_transform(test_img[i]) for i in range(len(test_img))])
    test_label = torch.Tensor(test_label).long()
    test_data = TensorDataset(test_img, test_label)

    # Generate candidate labelset
    labelset = generate_candidate_labelset(train_data=torch.tensor(data).permute(0, 3, 1, 2), train_labels=label, creation_method=creation_method, dataset_name='cifar100_trees', partial_rate=partial_rate, noisy_rate=noisy_rate)
    
    # Create augmented samples and packaged dataset
    partial_matrix_dataset = CIFAR100_Augmentention(data, labelset.float(), label.float())
    train_loader = DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    #### Calculate Jaccard similarity ####
    selected_files_labelset = torch.index_select(labelset, 0, torch.tensor(selected_files_idx))
    selected_files_truelabels = torch.index_select(label, 0, torch.tensor(selected_files_idx))
    logging.info('The 500 images in the forms:')
    logging.info('Average candidate num:{0}'.format(selected_files_labelset.sum(1).mean()))
    logging.info('The proportion of true labels in candidate label sets:{0}'.format(((torch.where(selected_files_labelset[torch.arange(len(selected_files_truelabels)), selected_files_truelabels]==1, True, False)==True).sum())/len(selected_files_truelabels)))

    # Calculate Jaccard similarity
    num_data = selected_files_truelabels.shape[0]
    num_class = len(torch.unique(selected_files_truelabels))
    labelset_reality = torch.zeros(num_data, num_class)
    for i in range(len(selected_files_truelabels)):
        unique_answer = df_form[i+1].unique()
        for value in unique_answer:
            labelset_reality[i, value] = 1
    
    jaccard_similarity = 0
    for i in range(num_data):
        set1 = set([i for i, val in enumerate(selected_files_labelset[i]) if val == 1])
        set2 = set([i for i, val in enumerate(labelset_reality[i]) if val == 1])
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_similarity += intersection / union if union != 0 else 0
    jaccard_similarity = jaccard_similarity/num_data
    logging.info('Jaccard similarity coefficient:{0}'.format(jaccard_similarity))

    return train_loader, test_loader

# CIFAR100_SM_500
def cifar100_small_mammals_part_dataloader(batch_size, creation_method='CLIG', partial_rate=0.1, noisy_rate=0.2):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    # Download dataset
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    df_form = pd.read_csv('dataset_collection/small_mammals_result.csv', header=None, index_col=0)

    # Find the index of 500 data in the form
    selected_files = df_form.loc['img'].tolist()
    df_form = df_form.drop(index=['img'])
    selected_files = [int(filename.split('.')[0]) for filename in selected_files]

    # Get only the data of small_mammals
    ## Training only gets the 500 data in the form
    ## Testing only gets the data of small_mammals
    train_img, train_label = [], []
    test_img, test_label = [], []
    selected_classes = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    for i in selected_files:
        select_img = train_data.data[i]
        select_label = train_data.targets[i]
        train_img.append(select_img)
        train_label.append(select_label)

    label_map = {}
    for subclass in selected_classes:
        subclass_idx = train_data.class_to_idx[subclass]
        label_map[subclass_idx] = len(label_map)
        test_idx = [i for i, label in enumerate(test_data.targets) if label == subclass_idx]
        for i in test_idx:
            select_img = test_data.data[i]
            select_label = test_data.targets[i]
            test_img.append(select_img)
            test_label.append(select_label)
    
    # Re-encode label to 0-4
    train_label = [label_map[train_label] for train_label in train_label]
    test_label = [label_map[test_label] for test_label in test_label]
    df_form = df_form.apply(lambda x: x.map(lambda y: int(y) if str(y).isdigit() else y))
    df_form = df_form.replace(label_map)

    # Adjust data type and format
    data = np.array(train_img)
    label = torch.Tensor(train_label).long()
    test_img = torch.stack([test_transform(test_img[i]) for i in range(len(test_img))])
    test_label = torch.Tensor(test_label).long()
    test_data = TensorDataset(test_img, test_label)
    
    # Make candidate labelset
    num_data = label.shape[0]
    num_class = len(torch.unique(label))
    labelset = torch.zeros(num_data, num_class)
    for i in range(len(label)):
        unique_answer = df_form[i+1].unique()
        for value in unique_answer:
            labelset[i, value] = 1
    
    # Create augmented samples and packaged dataset
    partial_matrix_dataset = CIFAR100_Augmentention(data, labelset.float(), label.float())
    train_loader = DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, test_loader

# CIFAR100_T_500
def cifar100_trees_part_dataloader(batch_size, creation_method='CLIG', partial_rate=0.1, noisy_rate=0.2):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    # Download dataset
    train_data = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    df_form = pd.read_csv('dataset_collection/trees_result.csv', header=None, index_col=0)

    # Find the index of 500 data in the form
    selected_files = df_form.loc['img'].tolist()
    df_form = df_form.drop(index=['img'])
    selected_files = [int(filename.split('.')[0]) for filename in selected_files]
    
    # Get only the data of trees
    ## Training only gets the 500 data in the form
    ## Testing only gets the data of trees
    train_img, train_label = [], []
    test_img, test_label = [], []
    selected_classes = ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree']
    label_map = {}
    for i in selected_files:
        select_img = train_data.data[i]
        select_label = train_data.targets[i]
        train_img.append(select_img)
        train_label.append(select_label)
    
    for subclass in selected_classes:
        subclass_idx = train_data.class_to_idx[subclass]
        label_map[subclass_idx] = len(label_map)
        test_idx = [i for i, label in enumerate(test_data.targets) if label == subclass_idx]
        for i in test_idx:
            select_img = test_data.data[i]
            select_label = test_data.targets[i]
            test_img.append(select_img)
            test_label.append(select_label)
    
    # Re-encode label to 0-4
    train_label = [label_map[train_label] for train_label in train_label]
    test_label = [label_map[test_label] for test_label in test_label]
    df_form = df_form.apply(lambda x: x.map(lambda y: int(y) if str(y).isdigit() else y))
    df_form = df_form.replace(label_map)
    
    # Adjust data type and format
    data = np.array(train_img)
    label = torch.Tensor(train_label).long()
    test_img = torch.stack([test_transform(test_img[i]) for i in range(len(test_img))])
    test_label = torch.Tensor(test_label).long()
    test_data = TensorDataset(test_img, test_label)

    # Make candidate labelset
    num_data = label.shape[0]
    num_class = len(torch.unique(label))
    labelset = torch.zeros(num_data, num_class)
    for i in range(len(label)):
        unique_answer = df_form[i+1].unique()
        for value in unique_answer:
            labelset[i, value] = 1
    
    # Create augmented samples and packaged dataset
    partial_matrix_dataset = CIFAR100_Augmentention(data, labelset.float(), label.float())
    train_loader = DataLoader(dataset=partial_matrix_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, test_loader

# Generate candidate labelset
def generate_candidate_labelset(train_data, train_labels, creation_method, dataset_name, partial_rate, noisy_rate):
    with torch.no_grad():
        num_data = train_labels.shape[0]
        num_class = len(torch.unique(train_labels))
        labelset = torch.zeros(num_data, num_class)
        logging.info('The method of generate candidate labelset: {0}'.format(creation_method))
        
        # Use APLG(partial_rate, noisy_rate) = APLG(gamma1, gamma2) to generate candidate labelset
        if creation_method=='APLG':
            partial_matrix = torch.ones(num_data, num_class) * partial_rate
            partial_matrix[torch.arange(num_data), train_labels] = 0
            for i in range(num_data):
                partial_labels_sampler = torch.distributions.Bernoulli(probs=partial_matrix[i])
                incorrect_labels_row = partial_labels_sampler.sample()
                labelset[i] = incorrect_labels_row.clone().detach()
            noisy_matrix = torch.ones(num_data, num_class) * (noisy_rate / (num_class-1))
            noisy_matrix[torch.arange(num_data), train_labels] = 1 - noisy_rate
            noisy_label_sampler = torch.distributions.Categorical(probs=noisy_matrix)
            noisy_labels = noisy_label_sampler.sample()
            labelset[torch.arange(num_data), noisy_labels] = 1
        
        # Use CLIG(delta1, delta2) to generate candidate labelset
        elif creation_method=='CLIG':
            # Obtain the prediction probability of each sample in each class by the model
            predict_matrix = get_predict_matrix(train_data, train_labels, dataset_name)

            # num_in_labelset = delta1, average labels per sample (excluding the true label)
            if dataset_name=='cifar10':
                num_in_labelset = 0.4572
            elif dataset_name=='cifar100_small_mammals':
                num_in_labelset = 1.3080
            elif dataset_name=='cifar100_trees':
                num_in_labelset = 1.1879
            
            noisy_matrix = torch.zeros(num_data, num_class)
            noisy_matrix[torch.arange(num_data), train_labels] = 1
            noisy_matrix1 = noisy_matrix * predict_matrix
            noisy_matrix2 = (1 - noisy_matrix) * ((1-predict_matrix[torch.arange(num_data), train_labels])/(num_class-1)).view(-1, 1)
            noisy_matrix = noisy_matrix1 + noisy_matrix2
            noisy_label_sampler = torch.distributions.Categorical(probs=noisy_matrix)
            noisy_labels = noisy_label_sampler.sample()
            labelset[torch.arange(num_data), noisy_labels] = 1

            partial_matrix = predict_matrix.clone().detach()
            partial_matrix[torch.arange(num_data), train_labels] = 0
            partial_matrix[torch.arange(num_data), noisy_labels] = 0
            flattened_partial_matrix = partial_matrix.view(-1)
            for i in range(int(num_in_labelset*num_data)):
                partial_labels_sampler = torch.distributions.Categorical(probs=flattened_partial_matrix)
                partial_labels = partial_labels_sampler.sample()
                row_index = partial_labels // num_class
                col_index = partial_labels % num_class
                labelset[row_index, col_index] = 1
                flattened_partial_matrix[partial_labels] = 0
                if i % 10000 == 0:
                    print('Completed generating {0} candidate label sets.'.format(i))
        
        logging.info('Finish Generating Candidate Label Sets!')
        logging.info('Average candidate num:{0}'.format(labelset.sum(1).mean()))
        logging.info('The proportion of true labels in candidate label sets:{0}'.format(((torch.where(labelset[torch.arange(num_data), train_labels]==1, True, False)==True).sum())/num_data))
        return labelset

# Obtain the prediction probability of each sample in each class by the model
def get_predict_matrix(train_data, train_labels, dataset_name):
    num_data = train_labels.shape[0]
    num_class = len(torch.unique(train_labels))
    
    # Adjust data type and format
    train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
    train_data = torch.stack([train_transform(train_data[i]) for i in range(len(train_data))])
    
    # load the trained model
    if dataset_name=='cifar10':
        model_filename = 'cifar10_20240308_015133.pt'
    elif dataset_name=='cifar100_small_mammals':
        model_filename = 'cifar100_SM_20240306_222558.pt'
    elif dataset_name=='cifar100_trees':
        model_filename = 'cifar100_T_20240306_223005.pt'
    model = resnet18_model(num_class)
    model.load_state_dict(torch.load('made_labelset_model/'+ model_filename))

    # Obtain the prediction probability by the model
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        predict_matrix = torch.zeros(num_data, num_class)
        batch_size = 64
        step = math.ceil(num_data / batch_size)
        for i in range(0, step):
            batch_data = train_data[i * batch_size : min((i + 1) * batch_size, num_data)].float().cuda()
            outputs = model(batch_data, eval_only=True)
            outputs_probas = torch.softmax(outputs, dim=1).clone().detach()
            predict_matrix[i * batch_size : min((i + 1) * batch_size, num_data)] = outputs_probas
            if i % 200 == 0:
                print('Completed obtaining {0} prediction probability.'.format(i))
        final_acc = accuracy(predict_matrix, train_labels)[0]
        logging.info('Accuracy of trained model is: {0}'.format(final_acc))
        return predict_matrix