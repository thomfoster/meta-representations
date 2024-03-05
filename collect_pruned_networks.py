import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
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
    prune_amount: float
    save_dir: str

    def asdict(self):
        return asdict(self) # this stops working inside ray


def collect_trained_network(config):
    run_name = petname.generate()
    model_fpath = config.save_dir + f'/{run_name}.pt'
    pruned_model_fpath = config.save_dir + f'/{run_name}-pruned.pt'
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
        parameters_to_prune = tuple([
            (module, 'weight')
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ])
        print(parameters_to_prune)
    elif config.architecture == 'cnn':
        parameters_to_prune = (
            (model.conv1, 'weight'),
            (model.conv2, 'weight'),
            (model.fc1, 'weight'),
            (model.fc2, 'weight'),
            (model.fc3, 'weight'),
        )
    else:
        raise RuntimeError()
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=config.prune_amount
    )

    # remove 'reparameterisation' and make pruning permanent
    for module, pname in parameters_to_prune:
        prune.remove(module, pname)

    accuracies = []
    for x, y_hat in test_loader:
        x, y_hat = x.to(DEVICE), y_hat.to(DEVICE)
        with torch.no_grad():
            y = model(x)
        pred = torch.argmax(y, dim=-1)
        accuracy = (y_hat == pred).to(torch.float).mean().cpu().item()
        accuracies.append(accuracy)
    pruned_acc = sum(accuracies)/len(accuracies)
    print(f'Pruned accuracy: {pruned_acc}')

    cd = config.__dict__
    cd['test_accuracy'] = acc
    cd['pruned_accuracy'] = pruned_acc
    with open(config_fpath, 'w') as f:
        f.write(json.dumps(cd))
    torch.save(model.state_dict(), pruned_model_fpath)
    print(f'Saved pruned to {model_fpath} and {config_fpath}')


if __name__ == '__main__':
    import random
    import ray
    import time

    N = 10_000

    start = time.time()

    ray.init()
    
    @ray.remote
    def call_with_random_config():
        for arch in ['mlp', 'cnn']:
            for dset in ['mnist', 'cifar']:
                config = Config(
                    architecture=arch,
                    dataset=dset,
                    optimiser=random.choice(['adam', 'sgd']),
                    batch_size=random.choice([2,4,8,16,32,64,128,256,512]),
                    learning_rate=random.choice([1e-1, 1e-2,1e-3,1e-4]),
                    epochs=random.choice([1,2,3,4]),
                    prune_amount=random.choice([0.01, 0.1, 0.2, 0.3, 0.4, 0.5]),
                    save_dir=f'./checkpoints/pruning_{arch}_{dset}'
                )
                collect_trained_network(config)

    handles = [call_with_random_config.remote() for i in range(N)]
    results = [ray.get(handle) for handle in handles]

    end = time.time()
    total = end - start

    print(f'Fimished in {total}s which is {total/N}s per examples')