import torch
from collections import defaultdict

class SAM:
    def __init__(self, optimizer, model, rho=0.05, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def perturb_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                self.state[p]["eps"] = torch.zeros_like(p)
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            self.state[p]["eps"] = eps
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def unperturb_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            # print(n)
            p.sub_(self.state[p]["eps"])

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()


