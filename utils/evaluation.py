from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Evaluator:
    def __init__(self, metrics):
        self.metrics = {}

        if isinstance(metrics, str):
            if metrics == 'default':
                metrics = ['Precision', 'Recall', 'F1Score']
            else:
                metrics = [metrics]

        for m in metrics:
            self.metrics[m] = globals()[m]()
        self.task_name = None
        self.scores = {'loss': 0}
        self.scores.update({m: None for m in list(self.metrics.keys())})
        self.iter = 0
        self.labels = {'y_true': [], 'y_pred': []}

    def prepare_task(self, task):
        self.reset()
        self.task_name = task['name']
        targets = list(task['map'].values())
        for m in self.metrics:
            self.metrics[m].set_targets(targets)

    def run(self, y_true, y_pred):
        return {name: m(y_true=y_true, y_pred=y_pred) for name, m in self.metrics.items()}

    def reset(self):
        self.scores = {'loss': 0}
        self.scores.update({m: None for m in list(self.metrics.keys())})
        self.iter = 0
        self.labels = {'y_true': [], 'y_pred': []}

    def update(self, y_true, y_pred, loss):
        self.scores['loss'] += loss.item()
        self.labels['y_true'] += y_true.clone().detach().cpu().tolist()
        self.labels['y_pred'] += y_pred.clone().detach().cpu().tolist()
        self.iter += 1

    def compute_scores(self):
        self.scores.update(self.run(y_true=self.labels['y_true'], y_pred=self.labels['y_pred']))
        self.scores['loss'] /= self.iter
        return self.scores


class BaseMetric:
    def __init__(self, targets=(0, 1)):
        self.targets = targets

    def set_targets(self, targets):
        self.targets = targets


class BaseAccuracy(BaseMetric):
    def __init__(self):
        super(BaseAccuracy, self).__init__()
        self.avg_mode = 'binary'

    def set_targets(self, targets):
        self.targets = targets
        if len(self.targets) == 2:
            self.avg_mode = 'binary'
        else:
            self.avg_mode = 'macro'


class Precision(BaseAccuracy):
    def __init__(self):
        super(Precision, self).__init__()

    def __call__(self, *args, **kwargs):
        return precision_score(y_true=kwargs['y_true'], y_pred=kwargs['y_pred'],
                               labels=self.targets, average=self.avg_mode, zero_division=0)


class Recall(BaseAccuracy):
    def __init__(self):
        super(Recall, self).__init__()

    def __call__(self, *args, **kwargs):
        return recall_score(y_true=kwargs['y_true'], y_pred=kwargs['y_pred'],
                            labels=self.targets, average=self.avg_mode, zero_division=0)


class F1Score(BaseAccuracy):
    def __init__(self):
        super(F1Score, self).__init__()

    def __call__(self, *args, **kwargs):
        return f1_score(y_true=kwargs['y_true'], y_pred=kwargs['y_pred'],
                        labels=self.targets, average=self.avg_mode, zero_division=0)


class Accuracy(BaseMetric):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, *args, **kwargs):
        return accuracy_score(y_true=kwargs['y_true'], y_pred=kwargs['y_pred'])
