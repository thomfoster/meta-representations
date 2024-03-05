import torch
import torch.nn as nn
import networks
import datasets
from tqdm import tqdm
import petname # for uuids
import json
from dataclasses import dataclass, asdict

DEVICE = torch.device('cpu')

@dataclass
class Config:
    architecture: str
    dataset: str
    optimiser: str
    batch_size: int
    learning_rate: float
    epochs: int
    save_dir: str


def collect_trained_network(config):
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
    print(f'Final accuracy: {acc}')
    
    run_name = petname.generate()
    model_fpath = config.save_dir + f'/{run_name}.pt'
    config_fpath = config.save_dir + f'/{run_name}.json'

    cd = config.__dict__
    cd['test_accuracy'] = acc
    with open(config_fpath, 'w') as f:
        f.write(json.dumps(cd))
    torch.save(model.state_dict(), model_fpath)
    print(f'Saved to {model_fpath} and {config_fpath}')


if __name__ == '__main__':
    import random
    import ray
    import time

    N = 20

    start = time.time()

    ray.init()
    
    @ray.remote
    def call():
        config = Config(
            architecture='cnn',
            dataset='mnist',
            optimiser='adam',
            batch_size=128,
            learning_rate=1e-5,
            epochs=0,
            save_dir='./checkpoints/identical_cnn_mnist_epoch_0'
        )
        collect_trained_network(config)

    handles = [call.remote() for i in range(N)]
    results = [ray.get(handle) for handle in handles]

    end = time.time()
    total = end - start

    print(f'Fimished in {total}s which is {total/N}s per examples')