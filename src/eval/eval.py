import numpy as np
import time
from tasks.task import get_task
from analyzer import Q

def eval(batches, render=False):
    try:
        simTime = time.time()
        pop, task, seeds, debug = batches
        env = get_task(task)
        fit = []
        pFit = []
        logs = []
        result = {}
        totalSteps = 0
        pop.compile()
        nn = pop.execute()
        nn.compile()
        for seed in seeds:
            env.seed(seed=seed)
            obs = env.reset()
            done = False
            totalReward = 0
            step = 0
            if (debug):
                log = ''
            while (not done):
                if (task in ('CartPole-v1', 'Retina', 'RetinaNM')):
                    action = nn.step(obs)[0]
                    if (action > 0):
                        action = 1
                    else:
                        action = 0
                else:
                    action = nn.step(obs) * env.action_space.high
                newObs, reward, done, info = env.step(action)
                totalReward += reward
                totalSteps += 1
                if (debug):
                    steplog = 'STEP: {0} \n INPUT: {1} \n OUTPUT: {2} \n NN VALUES: {3} \n REWARD: {4} \n'.format(
                        step, obs, action, nn.values, totalReward)
                    log += steplog
                    step += 1
                if (task == 'BipedalWalker-v3'):
                    if (max(abs(obs - newObs)) < 1e-5):
                        done = True
                obs = newObs
                if render:
                    env.render()
            fit.append(totalReward)
            if totalReward > 0:
                pFit.append(totalReward * (1 - 0.1 * nn.get_penalty()))
            else:
                pFit.append(totalReward)
            if (debug):
                logs.append(log)
            env.close()
        q, groups = Q(nn)
        pop.detach()
        simTime = time.time() - simTime
        result['fit'] = np.mean(fit)
        result['pfit'] = np.mean(pFit)
        result['logs'] = logs
        result['pop'] = pop
        result['Q'] = q
        result['steps'] = totalSteps
        result['time'] = simTime
        result['task'] = task
        result['seeds'] = seeds
    except OverflowError:
        print("OVERFLOW IN EVAL")
    return result

class EvalSummary():
    def __init__(self):
        self.store = {}
        self.metricName = ''

    def reduce(self, dicts, metricName):
        dicts = list(dicts)
        self.metricName = metricName
        keys = dicts[0].keys()
        for key in keys:
            self.store[key] = []
            for d in dicts:
                self.store[key].append(d[key])
        metrics = self.store[metricName]
        indices = np.flip(np.argsort(metrics)).tolist()
        for key in self.store.keys():
            arr = self.store[key]
            self.store[key] = [arr[i] for i in indices]


    def get_metric(self):
        return self.store[self.metricName]

    def get_res(self, name):
        return self.store[name]

    def summarise(self):
        self.store['best'] = self.store['pop'][0]
        index = np.argmax(self.store['time'])
        self.store['longest'] = (index, self.store['pop'][index])
        del self.store['pop']
