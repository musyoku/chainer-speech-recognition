import math
# for torch optim sgd
import numpy as np
from chainer import optimizers, cuda

# https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/tuner_utils/yellowfin.py

class YFOptimizer(object):
  def __init__(self, lr=0.1, momentum=0.0, clip_thresh=None, weight_decay=0.0,
    beta=0.999, curv_win_width=20, zero_debias=True, delta_mu=0.0):
    self.lr = lr
    self.momentum = momentum
    self.clip_threshold = clip_thresh
    self.beta = beta
    self.curv_win_width = curv_win_width
    self.zero_debias = zero_debias
    self.optimizer = optimizers.MomentumSGD(lr=lr, momentum=momentum)
    self.iter = 0

    # global states are the statistics
    self.global_state = {}

    # for decaying learning rate and etc.
    self.lr_factor = 1.0

  def set_lr_factor(self, factor):
    self.lr_factor = factor
    return

  def get_lr_factor(self):
    return self.lr_factor

  def zero_grad(self):
    self.optimizer.zero_grad()


  def zero_debias_factor(self):
    return 1.0 - self.beta ** (self.iter + 1)

  def curvature_range(self):
    global_state = self.global_state
    if self.iter == 0:
      global_state["curv_win"] = xp.zeros((self.curv_win_width, 1), dtype=xp.float32)
    curv_win = global_state["curv_win"]
    grad_norm_squared = self.global_state["grad_norm_squared"]
    curv_win[self.iter % self.curv_win_width] = grad_norm_squared
    valid_end = min(self.curv_win_width, self.iter + 1)
    beta = self.beta
    if self.iter == 0:
      global_state["h_min_avg"] = 0.0
      global_state["h_max_avg"] = 0.0
      self.h_min = 0.0
      self.h_max = 0.0
    global_state["h_min_avg"] = \
      global_state["h_min_avg"] * beta + (1 - beta) * torch.min(curv_win[:valid_end] )
    global_state["h_max_avg"] = \
      global_state["h_max_avg"] * beta + (1 - beta) * torch.max(curv_win[:valid_end] )
    if self.zero_debias:
      debias_factor = self.zero_debias_factor()
      self.h_min = global_state["h_min_avg"] / debias_factor
      self.h_max = global_state["h_max_avg"] / debias_factor
    else:
      self.h_min = global_state["h_min_avg"]
      self.h_max = global_state["h_max_avg"]
    return


  def grad_variance(self):
    global_state = self.global_state
    beta = self.beta
    self._grad_var = np.array(0.0, dtype=np.float32)
    for group in self.optimizer.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        state = self.optimizer.state[p]

        if self.iter == 0:
          state["grad_avg"] = grad.new().resize_as_(grad).zero_()
          state["grad_avg_squared"] = 0.0
        state["grad_avg"].mul_(beta).add_(1 - beta, grad)
        self._grad_var += torch.sum(state["grad_avg"] * state["grad_avg"] )
        
    if self.zero_debias:
      debias_factor = self.zero_debias_factor()
    else:
      debias_factor = 1.0

    self._grad_var /= -(debias_factor**2)
    self._grad_var += global_state['grad_norm_squared_avg'] / debias_factor
    return


  def dist_to_opt(self):
    global_state = self.global_state
    beta = self.beta
    if self.iter == 0:
      global_state["grad_norm_avg"] = 0.0
      global_state["dist_to_opt_avg"] = 0.0
    global_state["grad_norm_avg"] = \
      global_state["grad_norm_avg"] * beta + (1 - beta) * math.sqrt(global_state["grad_norm_squared"] )
    global_state["dist_to_opt_avg"] = \
      global_state["dist_to_opt_avg"] * beta \
      + (1 - beta) * global_state["grad_norm_avg"] / global_state['grad_norm_squared_avg']
    if self.zero_debias:
      debias_factor = self.zero_debias_factor()
      self._dist_to_opt = global_state["dist_to_opt_avg"] / debias_factor
    else:
      self._dist_to_opt = global_state["dist_to_opt_avg"]
    return


  def after_apply(self):
    # compute running average of gradient and norm of gradient
    beta = self.beta
    global_state = self.global_state
    if self.iter == 0:
      global_state["grad_norm_squared_avg"] = 0.0

    global_state["grad_norm_squared"] = 0.0
    for group in self.optimizer.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad.data
        # global_state['grad_norm_squared'] += torch.dot(grad, grad)
        global_state['grad_norm_squared'] += torch.sum(grad * grad)
        
    global_state['grad_norm_squared_avg'] = \
      global_state['grad_norm_squared_avg'] * beta + (1 - beta) * global_state['grad_norm_squared']
    # global_state['grad_norm_squared_avg'].mul_(beta).add_(1 - beta, global_state['grad_norm_squared'] )
        
    self.curvature_range()
    self.grad_variance()
    self.dist_to_opt()
    if self.iter > 0:
      self.get_mu()    
      self.get_lr()
      self.lr = beta * self.lr + (1 - beta) * self._lr_t
      self.momentum = beta * self.momentum + (1 - beta) * self._mu_t
    return


  def get_lr(self):
    self._lr_t = (1.0 - math.sqrt(self._mu_t) )**2 / self.h_min
    return


  def get_mu(self):
    coef = [-1.0, 3.0, 0.0, 1.0]
    coef[2] = -(3 + self._dist_to_opt**2 * self.h_min**2 / 2 / self._grad_var)
    roots = np.roots(coef)
    root = roots[np.logical_and(np.logical_and(np.real(roots) > 0.0, 
      np.real(roots) < 1.0), np.imag(roots) < 1e-5) ]
    assert root.size == 1
    dr = self.h_max / self.h_min
    self._mu_t = max(np.real(root)[0]**2, ( (np.sqrt(dr) - 1) / (np.sqrt(dr) + 1) )**2 )
    return 


  def update_hyper_param(self):
    for group in self.optimizer.param_groups:
      group['momentum'] = self.momentum
      group['lr'] = self.lr * self.lr_factor
    return


  def step(self):
    # apply update
    self.optimizer.step()

    # after appply
    self.after_apply()

    # update learning rate and momentum
    self.update_hyper_param()

    self.iter += 1
    return 
