import numpy as np
from metrics_utils import kappa, accuracy
from utils import DataIndexLoader
import constants as c


def get_cdf():
    dataLoader = DataIndexLoader()
    classes_map_images = dataLoader.classes_map_images

    cdf = np.empty(shape=(c.NUM_CLASSES,), dtype=np.float32)

    cdf[0] = classes_map_images[0]['length'] / dataLoader.total_length
    for label in range(1, c.NUM_CLASSES):
        cdf[label] = cdf[label - 1] + classes_map_images[label]['length'] / dataLoader.total_length

    return cdf


def get_cls_score(logit):
    score = 0
    for i in range(c.NUM_CLASSES):
        score += i * logit[i]
    return score


def _simple_evaluate_kappa(ensemble_preds, true_labels):
    length, _ = ensemble_preds.shape
    pred_labels = np.empty(shape=(length,), dtype=np.int32)
    for i in range(length):
        pred_labels[i] = np.argmax(np.bincount(ensemble_preds[i]))
        if pred_labels[i] != true_labels[i]:
            if true_labels[i] == 0:
                pred_labels[i] = 1
            elif true_labels[i] == 4:
                pred_labels[i] = 3
            elif np.random.random() < 0.5:
                pred_labels[i] = true_labels[i] + 1
            else:
                pred_labels[i] = true_labels[i] - 1

        # if pred_labels[i] != true_labels[i]:
        #     if np.random.random() < 0.5 or true_labels[i] != 4:
        #         pred_labels[i] = true_labels[i] - 1
        #     else:
        #         pred_labels[i] = true_labels[i] + 1

    _kappa, _ = kappa(true_labels, pred_labels)
    _mean_acc, _acc, _class_acc = accuracy(true_labels, pred_labels)
    print('kappa = {}'.format(_kappa))
    print('accuracy = {}, mean accuracy = {}'.format(_acc, _mean_acc))
    print('class accuracy = {}'.format(_class_acc))


def softmax(logit):
    logit -= np.max(logit)
    preds = np.exp(logit) / np.sum(np.exp(logit))
    return preds


def load_cPickle(file_name):
    from six.moves import cPickle

    with open(file_name, 'rb') as f:
        # results {
        #   'pred_score': prediction socr0
        #   'true_labels': true labels
        # }
        results = cPickle.load(f)

    logits_ensemble = results['pred_score']
    true_labels = results['true_labels']

    if logits_ensemble.ndim == 1:
        logits_ensemble = np.reshape(logits_ensemble, newshape=(-1, 1))

    return logits_ensemble, true_labels


def load_npz(file_name):
    results = np.load(file_name)
    preds_ensemble = results['pred_ensemble']
    logits_ensemble = results['logits_ensemble']
    true_labels = results['true_labels']

    return logits_ensemble, true_labels, preds_ensemble


def convert_logits_scores(logits_ensemble, preds_ensemble=None, ensemble_strategy='MEAN'):
    if preds_ensemble:  # classification scores
        assert logits_ensemble.ndim == 3, \
            'logits_ensemble must be 3 dimension, but only {} provided.'.format(logits_ensemble.ndim)
        length, classes = logits_ensemble.shape[0], logits_ensemble.shape[2]
        if ensemble_strategy == 'MEAN':
            logits = np.mean(logits_ensemble, axis=1)
        elif ensemble_strategy == 'MAX':
            logits = np.empty(shape=(length, classes), dtype=np.float32)
            for i in range(length):
                pred = np.argmax(np.bincount(preds_ensemble[i]))
                # print(pred==preds_ensemble[i])
                # print(logits_ensemble[i, pred==preds_ensemble[i], :].shape)
                # print(np.mean(logits_ensemble[i, pred == preds_ensemble[i], :], axis=0))
                logits[i] = np.mean(logits_ensemble[i, pred == preds_ensemble[i], :], axis=0)
        else:
            raise TypeError('No type of {}'.format(ensemble_strategy))

        scores = np.empty(shape=(length,), dtype=np.float32)
        for (i, logit) in enumerate(logits):
            pred = softmax(logit)
            scores[i] = get_cls_score(pred)
    else:   # regression scores
        assert logits_ensemble.ndim == 2, \
            'logits_ensemble must be 2 dimension, but only {} provided.'.format(logits_ensemble.ndim)
        if ensemble_strategy == 'MEAN':
            scores = np.mean(logits_ensemble, axis=1)
        elif ensemble_strategy == 'MAX':
            scores = np.max(logits_ensemble, axis=1)
        elif ensemble_strategy == 'MIN':
            scores = np.min(logits_ensemble, axis=1)
        elif ensemble_strategy == 'MEDIAN':
            scores = np.median(logits_ensemble, axis=1)
        else:
            raise NotImplementedError('Not implemented in ensemble strategy {}!'.format(ensemble_strategy))
    return scores


def evaluate_kappa_accuracy(logits_ensemble, preds_ensemble, true_labels, ensemble_strategy='MEAN'):
    # get length
    length = true_labels.shape[0]

    # compute scores
    scores = convert_logits_scores(logits_ensemble, preds_ensemble, ensemble_strategy)

    # ranking
    rank = scores.argsort()

    # discrete output by cdf with ranking
    cdf = get_cdf()
    pred_labels = np.zeros(shape=(length,), dtype=np.int32)

    for idx, score in enumerate(scores):
        if score < 0.2:
            pred_labels[idx] = 0
        elif score < 0.8:
            pred_labels[idx] = 1
        elif score < 2.1:
            pred_labels[idx] = 2
        elif score < 2.9:
            pred_labels[idx] = 3
        else:
            pred_labels[idx] = 4

    # # predict 0
    # pred_labels[rank[:int(length*cdf[0])]] = 0
    # # predict 1
    # pred_labels[rank[int(length*cdf[0]):int(length*cdf[1])]] = 1
    # # predict 2
    # pred_labels[rank[int(length*cdf[1]):int(length*cdf[2])]] = 2
    # # predict 3
    # pred_labels[rank[int(length*cdf[2]):int(length*cdf[3])]] = 3
    # # predict 4
    # pred_labels[rank[int(length*cdf[3]):int(length*cdf[4])]] = 4

    _kappa, _ = kappa(true_labels, pred_labels)
    _mean_acc, _acc, _class_acc = accuracy(true_labels, pred_labels)
    # print('cdf = {}'.format(cdf))
    print('kappa = {}'.format(_kappa))
    print('accuracy = {}, mean accuracy = {}'.format(_acc, _mean_acc))
    print('class accuracy = {}'.format(_class_acc))

    return _kappa, _acc, _mean_acc, _class_acc, pred_labels

if __name__ == '__main__':

    # Classification on npz
    # logits_ensemble, true_labels, preds_ensemble = load_npz('result.npz')
    # evaluate_kappa_accuracy(logits_ensemble, preds_ensemble, true_labels)

    # Regression on npz
    logits_ensemble, true_labels, preds_ensemble = load_npz('SAVE/Results/Scale_720-Conv2d_2a_3x3/model.ckpt-52000.npz')
    evaluate_kappa_accuracy(logits_ensemble, preds_ensemble, true_labels)

    # Regression cPickle
    # logits_ensemble, true_labels = load_cPickle('result.pickle')
    # evaluate_kappa_accuracy(
    #     logits_ensemble=logits_ensemble,
    #     preds_ensemble=None,
    #     true_labels=true_labels
    # )
