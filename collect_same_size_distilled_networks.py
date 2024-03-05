import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import networks
import datasets
from tqdm import tqdm
import petname # for uuids
import json
from dataclasses import dataclass, asdict

DEVICE = torch.device('mps')

@dataclass
class Config:
    architecture: str
    dataset: str
    optimiser: str
    batch_size: int
    learning_rate: float
    epochs: int
    temperature: float
    alpha: float
    save_dir: str


def collect_trained_network(config):
    run_name = petname.generate()
    model_fpath = config.save_dir + f'/{run_name}.pt'
    distilled_model_fpath = config.save_dir + f'/{run_name}-same-size-distilled.pt'
    config_fpath = config.save_dir + f'/{run_name}.json'

    if config.architecture == 'mlp':
        model = networks.MLP().to(DEVICE)
    elif config.architecture == 'cnn':
        model = networks.CNN().to(DEVICE)
    else:
        raise RuntimeError()

    if config.dataset == 'mnist':
        dataset = datasets.get_mnist()
    elif config.dataset == 'cifar':
        dataset = datasets.get_cifar()
    else:
        raise RuntimeError()

    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=config.batch_size, shuffle=True)

    if config.optimiser == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimiser == 'sgd':
        optimiser = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise RuntimeError()
    
    loss = nn.CrossEntropyLoss()
    prog_bar = tqdm(total=config.epochs * len(train_loader))

    for epoch in range(config.epochs):
        for step, (x, y_hat) in enumerate(train_loader):
            x, y_hat = x.to(DEVICE), y_hat.to(DEVICE)
            y = model(x)
            l = loss(y, y_hat)
            optimiser.zero_grad()
            l.backward()
            optimiser.step()
            prog_bar.update(1)
            if step % 100 == 0:
                prog_bar.set_description(f'training epoch: {epoch}/{config.epochs} loss: {l.detach().cpu().item()}')

    accuracies = []
    for x, y_hat in test_loader:
        x, y_hat = x.to(DEVICE), y_hat.to(DEVICE)
        with torch.no_grad():
            y = model(x)
        pred = torch.argmax(y, dim=-1)
        accuracy = (y_hat == pred).to(torch.float).mean().cpu().item()
        accuracies.append(accuracy)
    acc = sum(accuracies)/len(accuracies)
    print(f'Original trained accuracy: {acc}')
    torch.save(model.state_dict(), model_fpath)
    print(f'Saved original model to {model_fpath}')

    if config.architecture == 'mlp':
        student = networks.MLP().to(DEVICE)
    elif config.architecture == 'cnn':
        student = networks.CNN().to(DEVICE)
    else:
        raise RuntimeError()

    if config.optimiser == 'adam':
        optimiser = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    elif config.optimiser == 'sgd':
        optimiser = torch.optim.SGD(student.parameters(), lr=config.learning_rate)
    else:
        raise RuntimeError()
    
    prog_bar = tqdm(total=config.epochs * len(train_loader))
    for epoch in range(config.epochs):
        for step, (x, y_hat) in enumerate(train_loader):
            x, y_hat = x.to(DEVICE), y_hat.to(DEVICE)

            with torch.no_grad():
                teacher_logits = model(x)
                teacher_dist = F.softmax(teacher_logits / config.temperature)

            student_logits = student(x)
            student_dist = F.softmax(student_logits / config.temperature)

            l = config.alpha*loss(student_logits, y_hat) + (1-config.alpha)*loss(student_dist, teacher_dist)

            optimiser.zero_grad()
            l.backward()
            optimiser.step()
            prog_bar.update(1)
            if step % 100 == 0:
                prog_bar.set_description(f'epoch: {epoch} loss: {l.detach().cpu().item()}')

    accuracies = []
    for x, y_hat in test_loader:
        x, y_hat = x.to(DEVICE), y_hat.to(DEVICE)
        with torch.no_grad():
            y = student(x)
        pred = torch.argmax(y, dim=-1)
        accuracy = (y_hat == pred).to(torch.float).mean().cpu().item()
        accuracies.append(accuracy)
    distilled_acc = sum(accuracies)/len(accuracies)
    print(f'distilled accuracy: {distilled_acc}')

    cd = asdict(config)
    cd['test_accuracy'] = acc
    cd['distilled_acc'] = distilled_acc
    with open(config_fpath, 'w') as f:
        f.write(json.dumps(cd))
    torch.save(model.state_dict(), distilled_model_fpath)
    print(f'Saved distilled to {model_fpath} and {config_fpath}')


if __name__ == '__main__':
    config = Config(
        architecture='cnn',
        dataset='mnist',
        optimiser='adam',
        batch_size=128,
        learning_rate=1e-5,
        epochs=3,
        temperature=10,
        alpha=0.1,
        save_dir='./checkpoints'
    )
    collect_trained_network(config)