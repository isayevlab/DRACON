import numpy as np

from torch import nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score


def train_epoch(model, loader, optimizer, scheduler, clip_grad=1e-3):
    model.train()
    loss_list = []
    for batch in tqdm(loader):
        graph, targets = batch[0], batch[1:]
        optimizer.zero_grad()
        loss = model.get_loss(graph, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        loss_list.append(float(loss))
    return loss_list


def test(model, loader):
    sigmoid = nn.Sigmoid()
    model.eval()
    result = []
    for _ in range(model.n_model_heads):
        result.append({'target': [], 'predicted': []})

    for batch in loader:
        graph, targets = batch[0], batch[1:]
        outputs = model(graph)
        for i, (out, target) in enumerate(zip(outputs, targets)):
            predicted = (sigmoid(out) > .5).float().cpu().detach().numpy()
            target = target.float().cpu().detach().numpy()
            for pred, tar in zip(predicted, target):
                pred = pred[tar != -1]
                tar = tar[tar != -1]
                result[i]['target'].append(tar)
                result[i]['predicted'].append(pred)
    return result


def evaluate(model, loader, names=['Main product mapping', 'Detection of cenetr of reaction']):
    results = test(model, loader)
    scores = {}
    for key, result in enumerate(results):
        scores[key] = {'FM': [], 'F1': []}
        for tar, pred in zip(result['target'], result['predicted']):
            scores[key]['FM'].append(float(np.all(tar == pred)))
            scores[key]['F1'].append(f1_score(tar, pred))
        mean_f1 = np.mean(scores[key]['F1'])
        mean_fm = np.mean(scores[key]['FM'])
        scores[key]['F1_mean'] = mean_f1
        scores[key]['FM_mean'] = mean_fm
        print(f'{names[key]} was done with {mean_f1} F1-measure and {mean_fm} full-match accuracy')
    return scores

